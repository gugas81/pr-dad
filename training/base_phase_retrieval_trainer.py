import numpy as np
import os
import clearml
import tensorboardX
import torch
from pathlib import Path
from functools import partial
from datetime import datetime
from torch import Tensor
import torchvision
from torchvision import transforms
from torch.nn import functional as F
from training.dataset import create_data_loaders
from typing import Optional, Any, Dict, Union
import logging

from common import ConfigTrainer, set_seed, Losses, DataBatch, S3FileSystem
from common import im_concatenate, square_grid_im_concat, PATHS, im_save, fft2_from_rfft
from common import InferredBatch
import common.utils as utils
from training.augmentations import get_rnd_gauss_noise_like
from models.torch_dct import Dct2DForward

logging.basicConfig(level=logging.INFO)


class BaseTrainerPhaseRetrieval:
    _task = None
    _task_s3_path = None
    _log = logging.getLogger('BaseTrainerPhaseRetrieval')
    _s3 = S3FileSystem()
    _MODEL_CACHE_LOCAL_PATH = '/data/cache'

    def __init__(self, config: ConfigTrainer, experiment_name: Optional[str] = None):
        self._base_init()
        self._config: ConfigTrainer = config
        self._log.debug(f'Config params: \n {config} \n')

        if experiment_name is not None:
            self._init_experiment(experiment_name)

        self._global_step = 0
        self.device = 'cuda' if torch.cuda.is_available() and self._config.cuda else 'cpu'
        self.seed = self._config.seed
        self._fft_norm = self._config.fft_norm
        self._dbg_img_batch = self._config.dbg_img_batch
        self.log_interval = self._config.log_interval
        self.n_epochs = self._config.n_epochs_pr
        self.batch_size_train = self._config.batch_size_train
        self.batch_size_test = self._config.batch_size_test
        self.learning_rate = self._config.learning_rate
        self.img_size = config.image_size

        set_seed(self.seed)

        if self._config.use_dct_input:
            self.dct_input = Dct2DForward(utils.get_padded_size(self._config.image_size, self._config.add_pad))
        else:
            self.dct_input = None

        if self._config.debug_mode:
            torch.autograd.set_detect_anomaly(True)
            self._config.n_dataloader_workers = 0
            if self._task:
                self._task.add_tags(['DEBUG'])

        self._log.debug('init data loaders')
        self._init_data_loaders()

        self._init_dbg_data_batches()

    def _init_experiment(self, experiment_name: str):
        self._create_log_dir(experiment_name)
        self._create_loggers()
        self._init_trains(experiment_name)

    def _base_init(self):
        self._log = logging.getLogger(self.__class__.__name__)
        self._log.setLevel(logging.DEBUG)
        self._s3 = S3FileSystem()

    @classmethod
    def set_logger(cls):
        return logging.getLogger(__class__.__name__)

    def _init_dbg_data_batches(self):
        dbg_batch_tr = min(self.batch_size_train, self._dbg_img_batch)
        dbg_batch_ts = min(self.batch_size_test, self._dbg_img_batch)

        self.data_tr_batch = self.prepare_data_batch(iter(self.train_paired_loader).next(), is_train=True)
        self.data_ts_batch = self.prepare_data_batch(iter(self.test_loader).next(), is_train=False)

        self.data_tr_batch.image = self.data_tr_batch.image[:dbg_batch_tr]
        self.data_tr_batch.fft_magnitude = self.data_tr_batch.fft_magnitude[:dbg_batch_tr]
        self.data_tr_batch.label = self.data_tr_batch.label[:dbg_batch_tr]

        self.data_ts_batch.image = self.data_ts_batch.image[:dbg_batch_ts]
        self.data_ts_batch.fft_magnitude = self.data_ts_batch.fft_magnitude[:dbg_batch_ts]
        self.data_ts_batch.label = self.data_ts_batch.label[:dbg_batch_ts]

        if (self._config.gauss_noise is not None) and self._config.use_aug:
            self.data_tr_batch.image_noised = self.data_tr_batch.image_noised[:dbg_batch_tr]
            self.data_tr_batch.fft_magnitude_noised = self.data_tr_batch.fft_magnitude_noised[:dbg_batch_tr]
            self.data_ts_batch.image_noised = self.data_ts_batch.image_noised[:dbg_batch_ts]
            self.data_ts_batch.fft_magnitude_noised = self.data_ts_batch.fft_magnitude_noised[:dbg_batch_ts]

    def _init_data_loaders(self):
        self.train_paired_loader, self.train_unpaired_loader, self.test_loader, self.train_ds, self.test_ds = \
            create_data_loaders(config=self._config,
                                log=self._log,
                                s3=self._s3)

    def load_state(self, model_path: str) -> Dict[str, Any]:
        self._log.debug(f'Load state dict from: {model_path}')
        if model_path.startswith(self._s3.S3_CML_PATH):
            assert self._s3.isfile(model_path), f'not valid s3 path of model: {model_path}'
            rel_path_model = Path(model_path).relative_to(self._s3.S3_CML_PATH)
            local_model_path = os.path.join(self._MODEL_CACHE_LOCAL_PATH, rel_path_model)
            if os.path.isfile(local_model_path):
                self._log.debug(f'Exist cached model file in: {local_model_path} will be load from here')
            else:
                self._log.debug(f'Not Exist cached model file in: {local_model_path} will be cached here')
                os.makedirs(os.path.dirname(local_model_path), exist_ok=True)
                self._s3.download(model_path, local_model_path)
            loaded_sate = torch.load(local_model_path)

        elif self._s3.is_s3_url(model_path):
            assert self._s3.isfile(model_path), f'not valid s3 path of model: {model_path}'
            loaded_sate = self._s3.load_object(model_path, torch.load)
        else:
            assert os.path.isfile(model_path), f'not valid path of model: {model_path}'
            loaded_sate = torch.load(model_path)
        return loaded_sate

    def prepare_data_batch(self, data_batch: Dict[str, Any], is_train: bool = True) -> DataBatch:
        data_batch['is_paired'] = data_batch['is_paired'].cpu().numpy().all()
        data_batch: DataBatch = DataBatch.from_dict(data_batch).to(device=self.device)

        if torch.any(torch.isnan(data_batch.fft_magnitude)):
            data_batch.fft_magnitude = self.forward_magnitude_fft(data_batch.image)

        if (self._config.gauss_noise is not None) and self._config.use_aug:
            if is_train:
                img_noise = get_rnd_gauss_noise_like(data_batch.fft_magnitude,
                                                     self._config.gauss_noise,
                                                     p=self._config.prob_aug)

                data_batch.image_noised = data_batch.image + transforms.F.center_crop(img_noise,
                                                                                      data_batch.image.shape[-1])
                data_batch.fft_magnitude_noised = data_batch.fft_magnitude + self.forward_magnitude_fft(img_noise)
            else:
                data_batch.image_noised = data_batch.image
                data_batch.fft_magnitude_noised = data_batch.fft_magnitude

        return data_batch

    @staticmethod
    def load_config(config_obj: Union[str, dict], **kwargs) -> ConfigTrainer:
        if config_obj is None:
            config = ConfigTrainer
        elif isinstance(config_obj, str):
            config = BaseTrainerPhaseRetrieval.load_from_path(config_obj)
        elif isinstance(config_obj, dict):
            config = ConfigTrainer.from_dict(config_obj)
        config.log_path = PATHS.LOG
        config = config.update(**kwargs)
        return config

    @staticmethod
    def load_from_path(config_path: str) -> ConfigTrainer:
        s3 = S3FileSystem()
        if s3.is_s3_url(config_path):
            assert s3.isfile(config_path)
            config = s3.load_object(config_path, ConfigTrainer.from_data_file)
        else:
            assert os.path.exists(config_path)
            config = ConfigTrainer.from_data_file(config_path)
        return config

    def prepare_dbg_batch(self,
                          data_batch: Tensor,
                          num_images: Optional[int] = None,
                          image_size: Optional[int] = None,
                          grid_ch: bool = False) -> np.ndarray:
        def norm_img(img_arr):
            img_arr -= img_arr.min()
            img_arr /= max(img_arr.max(), np.finfo(img_arr.dtype).eps)
            return img_arr

        def resize_tensor(x: Tensor, images_size_: int) -> Tensor:
            assert x.ndimension() == 4
            x = x[:num_images]
            x = F.interpolate(x, (images_size_, images_size_), mode='bilinear', align_corners=False)
            return x

        if image_size:
            data_batch = resize_tensor(data_batch)

        num_images = min(num_images, self._config.batch_size_train) if num_images else self._config.batch_size_train

        data_batch_np = np.transpose(data_batch[: num_images].detach().cpu().numpy(), axes=(0, 2, 3, 1))
        data_batch_np = im_concatenate(data_batch_np)
        ch_size = data_batch_np.shape[-1]
        if ch_size > 3:
            data_batch_np = np.transpose(data_batch_np, axes=(2, 0, 1))
            data_batch_np = np.expand_dims(data_batch_np, -1)
            if grid_ch:
                data_batch_np = square_grid_im_concat(data_batch_np)
            else:
                data_batch_np = im_concatenate(data_batch_np, axis=0)

        data_batch_np = norm_img(data_batch_np)
        return data_batch_np

    def _create_loggers(self) -> None:
        self._tensorboard = tensorboardX.SummaryWriter(self._log_dir, flush_secs=60)
        self._tensorboard.add_hparams(self._config.as_dict(), metric_dict={}, global_step=0)

    def _init_trains(self, experiment_name: str) -> None:
        if self._config.use_tensor_board:
            time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            config_name = self._config.task_name.replace('_', '-').replace(' ', '-')
            ds_name = self._config.dataset_name.replace('_', '-').replace(' ', '-')
            task_name = f'{time_str}-{config_name}-{ds_name}-{experiment_name}'
            self._task = clearml.Task.init(task_name=task_name, project_name=self._config.project_name)
            self._task.connect_configuration(self._config.as_dict(), name='config_trainer')
            clearml.Logger.current_logger().set_default_upload_destination(PATHS.CLEARML_BUCKET)
            self._task.upload_artifact('config-trainer', self._config.as_dict())
        else:
            self._task = None

    def forward_magnitude_fft(self, data_batch: Tensor, is_noise: bool = False) -> Tensor:
        if (self._config.add_pad > 0.0) and (not is_noise):
            pad_value = utils.get_pad_val(self._config.image_size, self._config.add_pad)
            data_batch_pad = torchvision.transforms.functional.pad(data_batch, pad_value, padding_mode='edge')
        else:
            data_batch_pad = data_batch

        if self._config.use_dct_input:
            fft_data_batch = self.dct_input(data_batch_pad)
        elif self._config.use_rfft:
            fft_data_batch = torch.fft.rfft2(data_batch_pad, norm=self._config.fft_norm)
        else:
            fft_data_batch = torch.fft.fft2(data_batch_pad, norm=self._config.fft_norm)
        magnitude_batch = self._config.spectral_factor * torch.abs(fft_data_batch)
        return magnitude_batch

    def _log_images(self, image_batch: Tensor, step: int, tag_name: str, num_images: Optional[int] = None,
                    image_size: Optional[int] = None, grid_ch: bool = False) -> None:
        add_images_to_tensorboard = partial(self._tensorboard.add_images, global_step=step, dataformats='HWC')

        if isinstance(image_batch, Tensor):
            image_batch_np = self.prepare_dbg_batch(image_batch,
                                                    num_images=num_images,
                                                    image_size=image_size,
                                                    grid_ch=grid_ch)
        else:
            image_batch_np = image_batch
        add_images_to_tensorboard(tag_name, image_batch_np)
        task_s3_path = self.get_task_s3_path()
        if task_s3_path and self._s3.is_s3_url(task_s3_path):
            s3_img_path = os.path.join(task_s3_path, 'images', tag_name, f'{step}.png')
            self._s3.save_object(url=s3_img_path, saver=im_save, obj=image_batch_np)
            if isinstance(image_batch, Tensor):
                s3_img_tensors_path = os.path.join(task_s3_path, 'images-tensors', tag_name, f'{step}.png')
                self._s3.save_object(url=s3_img_tensors_path,
                                     saver=lambda path_: torchvision.utils.save_image(image_batch, path_))
        else:
            self._log.error(f'Non valid s3 path to save images: {task_s3_path}')

    def _grid_images(self, data_batch: DataBatch, inferred_batch: InferredBatch, normalize: bool = True) -> Tensor:
        inv_norm_transform = self.test_ds.get_inv_normalize_transform()
        img_grid = [inv_norm_transform(data_batch.image)]
        if (self._config.gauss_noise is not None) and self._config.use_aug:
            img_grid.append(inv_norm_transform(data_batch.image_noised))
        if inferred_batch.decoded_img is not None:
            img_grid.append(inv_norm_transform(inferred_batch.decoded_img))
        if inferred_batch.img_recon is not None:
            img_grid.append(inv_norm_transform(inferred_batch.img_recon))
        if inferred_batch.img_recon_ref is not None:
            img_grid.append(inv_norm_transform(inferred_batch.img_recon_ref))

        img_grid = torch.cat(img_grid, dim=-2)
        img_grid = torchvision.utils.make_grid(img_grid, normalize=normalize)
        return img_grid

    def _grid_diff_images(self, data_batch: DataBatch, inferred_batch: InferredBatch) -> Tensor:
        inv_norm_transform = self.test_ds.get_inv_normalize_transform()
        norm_orig_img = inv_norm_transform(data_batch.image)
        img_grid = [norm_orig_img]
        if (self._config.gauss_noise is not None) and self._config.use_aug:
            diff_decoded = torch.abs(norm_orig_img - inv_norm_transform(data_batch.image_noised))
            img_grid.append(diff_decoded)
        if inferred_batch.decoded_img is not None:
            diff_decoded = torch.abs(norm_orig_img - inv_norm_transform(inferred_batch.decoded_img))
            img_grid.append(diff_decoded)
        if inferred_batch.img_recon is not None:
            diff_recon = torch.abs(norm_orig_img - inv_norm_transform(inferred_batch.img_recon))
            img_grid.append(diff_recon)
        if inferred_batch.img_recon_ref is not None:
            diff_recon_ref = torch.abs(norm_orig_img - inv_norm_transform(inferred_batch.img_recon_ref))
            img_grid.append(diff_recon_ref)

        img_grid = torch.cat(img_grid, dim=-2)
        img_grid = torchvision.utils.make_grid(img_grid, normalize=False)
        return img_grid

    def _grid_fft_magnitude(self, data_batch: DataBatch, inferred_batch: InferredBatch) -> Tensor:
        def prepare_fft_img(fft_magnitude: Tensor) -> Tensor:
            if not self._config.use_dct_input:
                if self._config.use_rfft:
                    mag_size = utils.get_padded_size(self._config.image_size, self._config.add_pad)
                    fft_magnitude = fft2_from_rfft(fft_magnitude, (mag_size, mag_size))
                fft_magnitude = torch.fft.fftshift(fft_magnitude, dim=(-2, -1))

            if fft_magnitude.shape[-1] != self._config.image_size:
                fft_magnitude = F.interpolate(fft_magnitude, (self._config.image_size, self._config.image_size),
                                              mode='bilinear',
                                              align_corners=False)

            return fft_magnitude

        img_grid = []
        if data_batch.fft_magnitude is not None:
            img_grid.append(prepare_fft_img(data_batch.fft_magnitude))
        if (self._config.gauss_noise is not None) and self._config.use_aug:
            img_grid.append(prepare_fft_img(data_batch.fft_magnitude_noised))
        if inferred_batch.decoded_img is not None:
            fft_magnitude_ae_decoded = prepare_fft_img(self.forward_magnitude_fft(inferred_batch.decoded_img))
            img_grid.append(fft_magnitude_ae_decoded)
        if inferred_batch.img_recon is not None:
            fft_magnitude_recon = prepare_fft_img(self.forward_magnitude_fft(inferred_batch.img_recon))
            img_grid.append(fft_magnitude_recon)
        if inferred_batch.fft_magnitude_recon_ref is not None:
            img_grid.append(prepare_fft_img(inferred_batch.fft_magnitude_recon_ref))

        img_grid = torch.cat(img_grid, dim=-2)
        img_grid = torchvision.utils.make_grid(img_grid, normalize=False)
        return img_grid

    def _grid_features(self, inferred_batch: InferredBatch) -> (Tensor, Tensor):
        features_batch_enc = [inferred_batch.feature_recon[:self._config.dbg_features_batch]]
        features_batch_dec = []
        if inferred_batch.feature_recon_decoder is not None:
            features_batch_dec.append(inferred_batch.feature_recon_decoder[:self._config.dbg_features_batch])
        if inferred_batch.feature_encoder is not None:
            features_batch_enc.append(inferred_batch.feature_encoder[:self._config.dbg_features_batch])
        if inferred_batch.feature_decoder is not None:
            features_batch_dec.append(inferred_batch.feature_decoder[:self._config.dbg_features_batch])
        features_batch_enc = torch.cat(features_batch_enc, dim=-2)
        features_batch_dec = torch.cat(features_batch_dec, dim=-2)

        features_enc_grid = self._build_grid_features_map(features_batch_enc)
        features_dec_grid = self._build_grid_features_map(features_batch_dec)
        return features_enc_grid, features_dec_grid

    @staticmethod
    def _build_grid_features_map(features_batch: Tensor) -> Tensor:
        n_sqrt = int(np.sqrt(features_batch.shape[1]))
        features_grid = [torchvision.utils.make_grid(torch.unsqueeze(features, 1),
                                                     normalize=True, nrow=n_sqrt)[None]
                         for features in features_batch]
        features_grid = torchvision.utils.make_grid(torch.cat(features_grid))
        return features_grid

    def _save_img_to_s3(self, img: Tensor, path_url: str):
        if self._s3.is_s3_url(path_url):
            self._s3.save_object(url=path_url,
                                 saver=lambda path_: torchvision.utils.save_image(img, path_))

    def log_image_grid(self, image_grid: Optional[Tensor], tag_name: str, step: int):
        if image_grid is None:
            return
        self._tensorboard.add_images(tag=tag_name, img_tensor=image_grid, global_step=step, dataformats='CHW')
        if self._config.use_tensor_board:
            project_s3_path = self.get_task_s3_path()
            if project_s3_path and self._s3.is_s3_url(project_s3_path):
                s3_tensors_path = os.path.join(project_s3_path, 'images', tag_name, f'{step}.png')
                self._save_img_to_s3(image_grid, s3_tensors_path)
            else:
                self._log.error(f'Non valid s3 path to save images: {project_s3_path}')

    def _debug_images_grids(self, data_batch: DataBatch, inferred_batch: InferredBatch, normalize_img: bool = True):
        img_grid_grid = self._grid_images(data_batch, inferred_batch, normalize=normalize_img)
        img_diff_grid = self._grid_diff_images(data_batch, inferred_batch)
        fft_magnitude_grid_grid = self._grid_fft_magnitude(data_batch, inferred_batch)
        if self._config.predict_out == 'features':
            features_enc_grid, features_dec_grid = self._grid_features(inferred_batch)
        else:
            features_enc_grid, features_dec_grid = None, None
        return img_grid_grid, img_diff_grid, fft_magnitude_grid_grid, features_enc_grid, features_dec_grid

    def __del__(self):
        if self._task is not None:
            self._task.flush()
            self._task.close()

    def _create_log_dir(self, experiment_name: str) -> None:
        time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        log_dir_name = f'{time_str}_{experiment_name}'
        self._log_dir = os.path.join(self._config.log_path,
                                     self._config.task_name,
                                     self._config.dataset_name,
                                     log_dir_name)
        os.makedirs(self._log_dir, exist_ok=True)
        self.models_path = os.path.join(self._log_dir, 'models')
        os.makedirs(self.models_path, exist_ok=True)
        self._config.to_yaml(os.path.join(self._log_dir, 'config_params.yaml'))
        self._config.to_json(os.path.join(self._log_dir, 'config_params.json'))

    def _add_losses_tensorboard(self,  tag: str, losses: Losses, step: int = None) -> None:
        for metric_name, value in losses.__dict__.items():
            if value is not None:
                self._tensorboard.add_scalar(f"{metric_name}/{tag}", value.mean(), step)

    def get_task_name_id(self) -> str:
        return f'{self._task.name}.{self._task.task_id}'

    def get_task_s3_path(self) -> Optional[str]:
        if self._task_s3_path is None:
            self._set_task_s3_path()
        return self._task_s3_path

    def get_last_model_s3_path(self) -> Optional[str]:
        task_path = self.get_task_s3_path()
        if task_path:
            model_base_path = os.path.join(task_path, 'models', f'phase-retrieval-gan-model*.pt')
            model_paths = sorted(self._s3.glob(model_base_path))
            model_path = self._s3.s3_url(model_paths[-1]) if len(model_paths) > 0 else None
            return model_path
        else:
            return None

    def _set_task_s3_path(self):
        if self._task is not None:
            self._task_s3_path = os.path.join(self._task.logger.get_default_upload_destination(),
                                              self._task.get_project_name(),
                                              self.get_task_name_id())
            if not self._s3.exists(self._task_s3_path):
                self._log.error(f's3 path of project not exist: {self._task_s3_path}')
                self._task_s3_path = None
        else:
            self._task_s3_path = None

    def get_log(self) -> logging.Logger:
        return self._log
