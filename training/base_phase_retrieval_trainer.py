import numpy as np
import os
import clearml
import tensorboardX
import torch
from functools import partial
from datetime import datetime
from torch import Tensor
import torchvision
from torch.nn import functional as F
from training.dataset import create_data_loaders
from typing import Optional, Any, Dict
import logging

from common import ConfigTrainer, set_seed, Losses, DataBatch, S3FileSystem
from common import im_concatenate, square_grid_im_concat, PATHS, im_save
from common import InferredBatch


logging.basicConfig(level=logging.INFO)


class BaseTrainerPhaseRetrieval:
    _task = None

    def __init__(self, config: ConfigTrainer, experiment_name: str):
        self._config: ConfigTrainer = config
        self._global_step = 0
        self._log = logging.getLogger(self.__class__.__name__)
        self._log.setLevel(logging.DEBUG)
        self._s3 = S3FileSystem()
        self._create_log_dir(experiment_name)
        self._create_loggers()
        self._init_trains(experiment_name)

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

        self._log.debug(f'Config params: \n {config} \n')

        set_seed(self.seed)

        if self._config.debug_mode:
            torch.autograd.set_detect_anomaly(True)
            self._config.n_dataloader_workers = 0
            if self._task:
                self._task.add_tags(['DEBUG'])

        self._log.debug('init data loaders')
        self._init_data_loaders()

        self._init_dbg_data_batches()

    def _init_dbg_data_batches(self):
        dbg_batch_tr = min(self.batch_size_train, self._dbg_img_batch)
        dbg_batch_ts = min(self.batch_size_test, self._dbg_img_batch)

        self.data_tr_batch = self.prepare_data_batch(iter(self.train_paired_loader).next()).to(device=self.device)
        self.data_ts_batch = self.prepare_data_batch(iter(self.test_loader).next()).to(device=self.device)

        self.data_tr_batch.image = self.data_tr_batch.image[:dbg_batch_tr]
        self.data_tr_batch.fft_magnitude = self.data_tr_batch.fft_magnitude[:dbg_batch_tr]
        self.data_tr_batch.label = self.data_tr_batch.label[:dbg_batch_tr]

        self.data_ts_batch.image = self.data_ts_batch.image[:dbg_batch_ts]
        self.data_ts_batch.fft_magnitude = self.data_ts_batch.fft_magnitude[:dbg_batch_ts]
        self.data_ts_batch.label = self.data_ts_batch.label[:dbg_batch_ts]

    def _init_data_loaders(self):
        self.train_paired_loader, self.train_unpaired_loader, self.test_loader, self.train_ds, self.test_ds = \
            create_data_loaders(ds_name=self._config.dataset_name,
                                img_size=self.img_size,
                                use_aug=self._config.use_aug,
                                batch_size_train=self.batch_size_train,
                                batch_size_test=self.batch_size_test,
                                n_dataloader_workers=self._config.n_dataloader_workers,
                                paired_part=self._config.part_supervised_pairs,
                                fft_norm=self._fft_norm,
                                seed=self.seed,
                                log=self._log,
                                s3=self._s3)

    def load_state(self, model_path: str = None) -> Dict[str, Any]:
        if model_path is None:
            model_path = self._config.path_pretrained
        if self._s3.is_s3_url(model_path):
            loaded_sate = self._s3.load_object(model_path, torch.load)
        else:
            assert os.path.isfile(model_path)
            loaded_sate = torch.load(model_path)
        return loaded_sate

    def prepare_data_batch(self, item_data: Dict[str, Any]) -> DataBatch:
        is_paired = item_data['paired'].cpu().numpy().all()
        return DataBatch(image=item_data['image'].to(device=self.device),
                         fft_magnitude=item_data['fft_magnitude'].to(device=self.device),
                         label=item_data['label'].to(device=self.device),
                         is_paired=is_paired)

    @staticmethod
    def load_config(config_path,  **kwargs) -> ConfigTrainer:
        if config_path is None:
            config = ConfigTrainer()
        elif S3FileSystem.is_s3_url(config_path):
            config = S3FileSystem().load_object(config_path, ConfigTrainer.from_data_file)
        else:
            assert os.path.exists(config_path)
            config = ConfigTrainer.from_data_file(config_path)
        config.log_path = PATHS.LOG
        config = config.update(**kwargs)
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

    def forward_magnitude_fft(self, data_batch: Tensor) -> Tensor:
        fft_data_batch = torch.fft.fft2(data_batch, norm=self._config.fft_norm)
        magnitude_batch = torch.abs(fft_data_batch)
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
        if task_s3_path:
            s3_img_path = os.path.join(task_s3_path, 'images', tag_name, f'{step}.png')
            self._s3.save_object(url=s3_img_path, saver=im_save, obj=image_batch_np)
            if isinstance(image_batch, Tensor):
                s3_img_tensors_path = os.path.join(task_s3_path, 'images-tensors', tag_name, f'{step}.png')
                self._s3.save_object(url=s3_img_tensors_path,
                                     saver=lambda path_: torchvision.utils.save_image(image_batch, path_))

    def _grid_images(self, data_batch: DataBatch, inferred_batch: InferredBatch, normalize: bool = True) -> Tensor:
        inv_norm_transform = self.test_ds.get_inv_normalize_transform()
        img_grid = [inv_norm_transform(data_batch.image)]
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
        img_grid = []
        if data_batch.fft_magnitude is not None:
            img_grid.append(torch.fft.fftshift(data_batch.fft_magnitude, dim=(-2, -1)))
        if inferred_batch.decoded_img is not None:
            fft_magnitude_ae_decoded = torch.fft.fftshift(self.forward_magnitude_fft(inferred_batch.decoded_img),
                                                          dim=(-2, -1))
            img_grid.append(fft_magnitude_ae_decoded)
        if inferred_batch.img_recon is not None:
            fft_magnitude_recon = torch.fft.fftshift(self.forward_magnitude_fft(inferred_batch.img_recon), dim=(-2, -1))
            img_grid.append(fft_magnitude_recon)
        if inferred_batch.fft_magnitude_recon_ref is not None:
            img_grid.append(torch.fft.fftshift(inferred_batch.fft_magnitude_recon_ref, dim=(-2, -1)))

        img_grid = torch.cat(img_grid, dim=-2)
        img_grid = torchvision.utils.make_grid(img_grid, normalize=False)
        return img_grid

    def _grid_features(self, inferred_batch: InferredBatch) -> Tensor:
        features_batch = [inferred_batch.feature_recon[:self._config.dbg_features_batch]]
        if inferred_batch.feature_encoder is not None:
            features_batch.append(inferred_batch.feature_encoder[:self._config.dbg_features_batch])
        features_batch = torch.cat(features_batch, dim=-2)

        n_sqrt = int(np.sqrt(features_batch.shape[1]))
        features_grid = [torchvision.utils.make_grid(torch.unsqueeze(features, 1),
                                                     normalize=True, nrow=n_sqrt)[None]
                         for features in features_batch]
        features_grid = torchvision.utils.make_grid(torch.cat(features_grid))
        return features_grid

    def _save_img_to_s3(self, img: Tensor, path_url):
        self._s3.save_object(url=path_url,
                             saver=lambda path_: torchvision.utils.save_image(img, path_))

    def log_image_grid(self, image_grid: Tensor, tag_name: str, step: int):
        s3_tensors_path = os.path.join(self.get_task_s3_path(), 'images', tag_name, f'{step}.png')
        self._tensorboard.add_images(tag=tag_name, img_tensor=image_grid, global_step=step, dataformats='CHW')
        self._save_img_to_s3(image_grid, s3_tensors_path)

    def _debug_images_grids(self, data_batch: DataBatch, inferred_batch: InferredBatch, normalize_img: bool = True):
        img_grid_grid = self._grid_images(data_batch, inferred_batch, normalize=normalize_img)
        img_diff_grid = self._grid_diff_images(data_batch, inferred_batch)
        fft_magnitude_grid_grid = self._grid_fft_magnitude(data_batch, inferred_batch)
        features_grid_grid = self._grid_features(inferred_batch)
        return img_grid_grid, img_diff_grid, fft_magnitude_grid_grid, features_grid_grid

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
        path_task = os.path.join(self._task.logger.get_default_upload_destination(),
                                 self._task.get_project_name(),
                                 self.get_task_name_id())
        path_task = path_task if self._s3.exists(path_task) else None
        return path_task

    def get_log(self) -> logging.Logger:
        return self._log
