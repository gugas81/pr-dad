import numpy as np
import torch.nn as nn
import os
import clearml
import tensorboardX
import torch
import torchvision
from functools import partial
from datetime import datetime
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F
import logging
from common import TensorBatch, ConfigTrainer, set_seed, Losses
from common import im_concatenate, square_grid_im_concat
from typing import Optional

DATASETS_PATH = './files/'
CLEARML_BUCKET = 's3://data-clearml'

logging.basicConfig(level=logging.INFO)


class TrainerPhaseRetrieval:
    _task = None

    def __init__(self, config: ConfigTrainer, experiment_name: str):
        self.config: ConfigTrainer = config
        self._global_step = 0
        self._log = logging.getLogger(self.__class__.__name__)
        self._log.setLevel(logging.DEBUG)
        self._create_log_dir(experiment_name)
        self._create_loggers()
        self._init_trains(experiment_name)

        self.device = 'cuda' if torch.cuda.is_available() and self.config.cuda else 'cpu'
        self.seed = self.config.seed
        self._fft_norm = self.config.fft_norm
        self._dbg_img_batch = self.config.dbg_img_batch
        self.log_interval = self.config.log_interval
        self.n_epochs = self.config.n_epochs_pr
        self.batch_size_train = self.config.batch_size_train
        self.batch_size_test = self.config.batch_size_test
        self.learning_rate = self.config.learning_rate
        self.img_size = config.image_size

        self._log.debug(f'Config params: {config}')

        set_seed(self.seed)

        if self.config.debug_mode:
            torch.autograd.set_detect_anomaly(True)
            self.config.n_dataloader_workers = 0
            if self._task:
                self._task.add_tags(['DEBUG'])

        self._init_data_loaders()

        self._init_dbg_data_batches()

    def _init_dbg_data_batches(self):
        self.data_tr_batch = iter(self.train_loader).next()[0].to(device=self.device)
        self.data_ts_batch = iter(self.test_loader).next()[0].to(device=self.device)
        self.data_tr_batch = self.data_tr_batch[:min(self.batch_size_train, self._dbg_img_batch)]
        self.data_ts_batch = self.data_ts_batch[:min(self.batch_size_test, self._dbg_img_batch)]

    def _init_data_loaders(self):
        data_transforms = [transforms.Resize((self.img_size, self.img_size))]
        ds_name = self.config.dataset_name.lower()
        image_dataset = None
        if ds_name == 'mnist':
            image_dataset = torchvision.datasets.MNIST
        elif ds_name == 'emnist':
            image_dataset = partial(torchvision.datasets.EMNIST, split='balanced')
        elif ds_name == 'fashion-mnist':
            image_dataset = torchvision.datasets.FashionMNIST
        elif ds_name == 'celeba':
            image_dataset = torchvision.datasets.CelebA
            data_transforms.append(transforms.Grayscale(num_output_channels=1))
        else:
            raise NameError(f'Not valid ds type {ds_name}')
        # data_transforms.append(transforms.Resize((self.img_size, self.img_size)))
        if self.config.use_aug:
            data_transforms_aoug = [transforms.RandomRotation(90.0, interpolation=InterpolationMode.BILINEAR),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip()]
            data_transforms += data_transforms_aoug
        data_transforms = data_transforms + [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        data_transforms = transforms.Compose(data_transforms)
        ds_path = os.path.join(DATASETS_PATH, ds_name)
        self.train_dataset = image_dataset(root=ds_path, train=True, download=True, transform=data_transforms)
        self.test_dataset = image_dataset(root=ds_path, train=False, download=True, transform=data_transforms)
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.batch_size_train,
                                       shuffle=True,
                                       worker_init_fn=np.random.seed(self.seed),
                                       num_workers=self.config.n_dataloader_workers)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.batch_size_test,
                                      shuffle=False,
                                      worker_init_fn=np.random.seed(self.seed),
                                      num_workers=self.config.n_dataloader_workers)

    def forward_magnitude_fft(self, data_batch: Tensor) -> Tensor:
        fft_data_batch = torch.fft.fft2(data_batch, norm=self._fft_norm)
        magnitude_batch = torch.abs(fft_data_batch)
        return magnitude_batch

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

        num_images = min(num_images, self.config.batch_size_train) if num_images else self.config.batch_size_train

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
        self._tensorboard.add_hparams(self.config.as_dict(), metric_dict={}, global_step=0)

    def _init_trains(self, experiment_name: str) -> None:
        if self.config.use_tensor_board:
            time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            config_name = self.config.task_name.replace('_', '-').replace(' ', '-')
            ds_name = self.config.dataset_name.replace('_', '-').replace(' ', '-')
            task_name = f'{time_str}-{config_name}-{ds_name}-{experiment_name}'
            self._task = clearml.Task.init(task_name=task_name, project_name=self.config.project_name)
            self._task.connect_configuration(self.config.as_dict(), name='config_trainer')
            clearml.Logger.current_logger().set_default_upload_destination(CLEARML_BUCKET)
            self._task.upload_artifact('config-trainer', self.config.as_dict())
        else:
            self._task = None

    def _log_images(self, image_batch: Tensor, step: int, tag_name: str, num_images: Optional[int] = None,
                    image_size: Optional[int] = None, grid_ch: bool = False) -> None:
        add_images = partial(self._tensorboard.add_images, global_step=step, dataformats='HWC')

        if isinstance(image_batch, Tensor):
            image_batch_np = self.prepare_dbg_batch(image_batch,
                                                    num_images=num_images,
                                                    image_size=image_size,
                                                    grid_ch=grid_ch)
        else:
            image_batch_np = image_batch
        add_images(tag_name, image_batch_np)

    def __del__(self):
        if self._task is not None:
            self._task.flush()
            self._task.close()

    def _create_log_dir(self, experiment_name: str) -> None:
        time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        log_dir_name = f'{time_str}_{experiment_name}'
        self._log_dir = os.path.join(self.config.log_path,
                                     self.config.task_name,
                                     self.config.dataset_name,
                                     log_dir_name)
        os.makedirs(self._log_dir, exist_ok=True)
        self.save_path = os.path.join(self._log_dir, 'models')
        os.makedirs(self.save_path, exist_ok=True)
        self.config.to_yaml(os.path.join(self._log_dir, 'config_params.yaml'))
        self.config.to_json(os.path.join(self._log_dir, 'config_params.json'))

    def _add_losses_tensorboard(self,  tag: str, losses: Losses, step: int = None) -> None:
        for metric_name, value in losses.__dict__.items():
            if value:
                self._tensorboard.add_scalar(f"{metric_name}/{tag}", value.mean(), step)

