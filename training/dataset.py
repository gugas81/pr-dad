import logging
import os
from functools import partial
from typing import Union, Tuple, Dict, Any, Optional
import fire
import numpy as np
import torch
import torchvision
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from common import PATHS, S3FileSystem


class PhaseRetrievalDataset(Dataset):
    def __init__(self, ds_name: str, img_size: int, train: bool, use_aug: bool,
                 paired_part: float, fft_norm: str, log: logging.Logger, seed: int, s3: Optional[S3FileSystem] = None):
        np.random.seed(seed=seed)

        self._ds_name = ds_name
        self.img_size = img_size
        self._fft_norm = fft_norm
        self._img_size = img_size
        self._is_train = train
        self._use_aug = use_aug
        self._log = log
        self._s3 = S3FileSystem() if s3 is None else s3

        data_transforms = [transforms.Resize((self.img_size, self.img_size))]

        ds_name = ds_name.lower()
        if ds_name == 'mnist':
            ds_type = torchvision.datasets.MNIST
        elif ds_name == 'emnist':
            ds_type = partial(torchvision.datasets.EMNIST, split='balanced')
        elif ds_name == 'fashion-mnist':
            ds_type = torchvision.datasets.FashionMNIST
        elif ds_name == 'celeba':
            ds_type = lambda root, train, download, transform:  torchvision.datasets.CelebA(root=root,
                                                                                            split='train' if self._is_train else 'test',
                                                                                            download=download,
                                                                                            transform=transform)
            transform_celeba = transforms.Compose([
                                               transforms.Resize(self.img_size),
                                               transforms.CenterCrop(self.img_size),

                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),),
                                               transforms.Grayscale(num_output_channels=1),
                                           ])

            data_transforms.append(transforms.Grayscale(num_output_channels=1))
        else:
            raise NameError(f'Not valid ds type {ds_name}')

        if self._use_aug:
            data_transforms_aoug = [transforms.RandomRotation(90.0, interpolation=InterpolationMode.BILINEAR),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip()]
            data_transforms += data_transforms_aoug
        data_transforms = data_transforms + [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        data_transforms = transforms.Compose(data_transforms)
        ds_path = os.path.join(PATHS.DATASETS, ds_name)

        if ds_name == 'celeba':
            s3_ds_path = os.path.join(PATHS.DATASETS_S3, ds_name)
            if not os.path.exists(os.path(ds_path, ds_name)):
                self._log.debug(f'ds {ds_name} not exist in local path: {ds_path}, download from {s3_ds_path}')
                s3.download(rpath=s3_ds_path, lpath=ds_path, recursive=True)
            data_transforms = transform_celeba

        self.image_dataset = ds_type(root=ds_path,
                                     train=self._is_train,
                                     download=True,
                                     transform=data_transforms)
        self.len_ds = len(self.image_dataset)
        if paired_part < 1.0:
            paired_len = int(paired_part * self.len_ds)
            inds = np.random.permutation(self.len_ds)
            self.paired_ind = inds[:paired_len]
            self.unpaired_paired_ind = inds[paired_len:]
        else:
            self.paired_ind = list(range(self.len_ds))
            self.unpaired_paired_ind = []

    def __len__(self):
        return self.len_ds

    def __getitem__(self, idx: Union[int, Tuple[int, int, int]]) -> Dict[str, Any]:
        while True:
            try:
                return self._get_item(idx)
            except Exception as e:
                self._log.error(f'Error loading item {idx} from the dataset: {e}')

    def _get_item(self, idx: Union[int, Tuple[int, int, int]]) -> Dict[str, Tensor]:
        is_paired = idx in self.paired_ind
        image_data = self.image_dataset[idx][0]
        label = self.image_dataset[idx][1]
        fft_magnitude = self._forward_magnitude_fft(image_data)
        if not is_paired:
            image_data = self.image_dataset[np.random.choice(self.paired_ind)][0]
        item = {'image': image_data, 'fft_magnitude': fft_magnitude, 'label': label, 'paired': is_paired}
        return item

    def _forward_magnitude_fft(self, image_data: Tensor) -> Tensor:
        fft_data_batch = torch.fft.fft2(image_data, norm=self._fft_norm)
        fft_magnitude = torch.abs(fft_data_batch)
        return fft_magnitude


def create_data_loaders(ds_name: str, img_size: int, use_aug: bool, batch_size_train: int, batch_size_test: int,
                        seed: int,
                        n_dataloader_workers: int, paired_part: float, fft_norm: str, log: logging.Logger,
                        s3: Optional[S3FileSystem] = None):
    train_dataset = PhaseRetrievalDataset(ds_name=ds_name, img_size=img_size, train=True, use_aug=use_aug,
                                          paired_part=paired_part, fft_norm=fft_norm, log=log, seed=seed, s3=s3)

    test_dataset = PhaseRetrievalDataset(ds_name=ds_name, img_size=img_size, train=False, use_aug=use_aug,
                                         paired_part=1.0, fft_norm=fft_norm, log=log, seed=seed, s3=s3)

    paired_tr_sampler = torch.utils.data.SubsetRandomSampler(train_dataset.paired_ind)
    unpaired_tr_sampler = torch.utils.data.SubsetRandomSampler(train_dataset.unpaired_paired_ind)

    train_paired_loader = DataLoader(train_dataset, batch_size=batch_size_train,
                                     worker_init_fn=np.random.seed(seed), num_workers=n_dataloader_workers,
                                     sampler=paired_tr_sampler)

    train_unpaired_loader = DataLoader(train_dataset, batch_size=batch_size_train,
                                       worker_init_fn=np.random.seed(seed), num_workers=n_dataloader_workers,
                                       sampler=unpaired_tr_sampler)

    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False,
                             worker_init_fn=np.random.seed(seed),
                             num_workers=n_dataloader_workers)

    return train_paired_loader, train_unpaired_loader, test_loader


def example_mnist_upiared():
    logging.basicConfig(level=logging.DEBUG)
    log = logging.getLogger('exmaple_mnist_upiared')
    train_paired_loader, train_unpaired_loader, test_loader = create_data_loaders(ds_name='mnist',
                                                                                  img_size=32,
                                                                                  use_aug=False,
                                                                                  batch_size_train=128,
                                                                                  batch_size_test=256,
                                                                                  n_dataloader_workers=0,
                                                                                  paired_part=1.0,
                                                                                  fft_norm='ortho',
                                                                                  seed=1,
                                                                                  log=log)
    from itertools import chain, islice

    def shuffle(generator, buffer_size):
        while True:
            buffer = list(islice(generator, buffer_size))
            if len(buffer) == 0:
                break
            np.random.shuffle(buffer)
            for item in buffer:
                yield item

    log.debug('train_loader')
    for data_batch in chain(train_paired_loader, train_unpaired_loader):
        log.debug(data_batch['image'].shape)
        log.debug(data_batch['paired'].numpy().all())


if __name__ == '__main__':
    fire.Fire()
