import logging
import os
from functools import partial
from typing import Union, Tuple, Dict, Any, Optional, Callable, Sequence
import fire
import numpy as np
import torch
import torchvision
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from training.augmentations import RandomGammaCorrection
from common import PATHS, S3FileSystem, NormalizeInverse, DataBatch


class PhaseRetrievalDataset(Dataset):
    def __init__(self, ds_name: str, img_size: int, train: bool, use_aug: bool, rot_degrees: float, is_gan: bool,
                 paired_part: float, fft_norm: str, log: logging.Logger, seed: int, s3: Optional[S3FileSystem] = None,
                 use_rfft: bool = False,
                 prob_aug: float = 1.0,
                 gamma_corr: Sequence[float] = (0.85, 1.125),
                 gauss_blur: Sequence[float] = (0.5, 1.5),
                 sharpness_factor: float = 1.0,
                 rnd_vert_flip: bool = False,
                 rnd_horiz_flip: bool = False

                 ):
        def celeba_ds(root: str, train: bool, download: bool, transform: Optional[Callable] = None):
            return torchvision.datasets.CelebA(root=root,
                                               split='train' if train else 'test',
                                               download=download,
                                               transform=transform)

        np.random.seed(seed=seed)

        self._ds_name = ds_name
        self.img_size = img_size
        self._fft_norm = fft_norm
        self._img_size = img_size
        self._is_train = train
        self._is_gan = is_gan
        self._use_aug = use_aug
        self._use_rfft = use_rfft
        self._log = log
        self._s3 = S3FileSystem() if s3 is None else s3

        ds_name = ds_name.lower()
        ds_path = os.path.join(PATHS.DATASETS, ds_name)
        alignment_transform = transforms.Resize(self.img_size)
        self.norm_mean = 0.1307
        self.norm_std = 0.3081
        normalize_transform = transforms.Normalize((self.norm_mean,), (self.norm_std,))
        is_rgb = False
        if ds_name == 'mnist':
            ds_class = torchvision.datasets.MNIST
        elif ds_name == 'emnist':
            ds_class = partial(torchvision.datasets.EMNIST, split='balanced')
        elif ds_name == 'fashion-mnist':
            ds_class = torchvision.datasets.FashionMNIST
        elif ds_name == 'kmnist':
            ds_class = torchvision.datasets.KMNIST
        elif ds_name == 'celeba':
            self._download_ds_from_s3(ds_name, ds_path)
            self.norm_mean = 0.5
            self.norm_std = 0.5
            is_rgb = True
            ds_class = celeba_ds
            alignment_transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.CenterCrop(self.img_size)])
            normalize_transform = transforms.Normalize((self.norm_mean, self.norm_mean, self.norm_mean),
                                                       (self.norm_std, self.norm_std, self.norm_std))
        else:
            raise NameError(f'Not valid ds type {ds_name}')

        data_transforms = [alignment_transform, transforms.ToTensor(), normalize_transform]
        if is_rgb:
            data_transforms.append(transforms.Grayscale(num_output_channels=1))

        if self._use_aug:
            augmentations_transforms = []
            if sharpness_factor is not None:
                augmentations_transforms.append(transforms.RandomAdjustSharpness(sharpness_factor=1.0, p=prob_aug))

            if gauss_blur is not None:
                augmentations_transforms.append(transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=gauss_blur)], p=prob_aug))

            if gamma_corr is not None:
                augmentations_transforms.append(transforms.RandomApply([
                    RandomGammaCorrection(gamma_range=gamma_corr)], p=prob_aug))

            if len(augmentations_transforms) > 0:
                augmentations_transforms = [transforms.RandomOrder(augmentations_transforms)]

            if rnd_vert_flip:
                augmentations_transforms.append(transforms.RandomVerticalFlip(p=prob_aug))

            if rnd_horiz_flip:
                augmentations_transforms.append(transforms.RandomHorizontalFlip(p=prob_aug))

            if rot_degrees > 0.0:
                augmentations_transforms = [transforms.RandomRotation(rot_degrees,
                                                                      interpolation=InterpolationMode.BILINEAR)] + \
                                           augmentations_transforms

            if isinstance(augmentations_transforms, list) and len(augmentations_transforms) > 0:
                augmentations_transforms = transforms.Compose(augmentations_transforms)
                data_transforms.append(augmentations_transforms)

        if isinstance(data_transforms, list) and len(data_transforms) > 0:
            data_transforms = transforms.Compose(data_transforms)

        self.image_dataset = ds_class(root=ds_path,
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

    def _download_ds_from_s3(self, ds_name, ds_path):
        s3_ds_path = os.path.join(PATHS.DATASETS_S3, ds_name)
        if not os.path.exists(os.path.join(ds_path, ds_name)):
            self._log.debug(f'ds {ds_name} not exist in local path: {ds_path}, download from {s3_ds_path}')
            self._s3.download(rpath=s3_ds_path, lpath=ds_path, recursive=True)

    def __len__(self):
        return self.len_ds

    def __getitem__(self, idx: Union[int, Tuple[int, int, int]]) -> Dict[str, Any]:
        while True:
            try:
                item_batch = self._get_item(idx)
                return item_batch.as_dict()
            except Exception as e:
                self._log.error(f'Error loading item {idx} from the dataset: {e}')

    def _get_item(self, idx: Union[int, Tuple[int, int, int]]) -> DataBatch:
        is_paired = idx in self.paired_ind
        img_item = self.image_dataset[idx]
        image_data = img_item[0].to(device='cpu')
        label = img_item[1].to(device='cpu')
        fft_magnitude = self._forward_magnitude_fft(image_data)
        if not is_paired:
            image_data = self.image_dataset[np.random.choice(self.paired_ind)][0]
        item_batch = DataBatch(image=image_data, fft_magnitude=fft_magnitude, label=label, is_paired=is_paired)
        if self._is_gan:
            img_discrim_item = self.image_dataset[np.random.choice(self.paired_ind)]
            item_batch.image_discrim = img_discrim_item[0]
            item_batch.label_discrim = img_discrim_item[1]
        return item_batch

    def _forward_magnitude_fft(self, image_data: Tensor) -> Tensor:
        if self._use_rfft:
            fft_data_batch = torch.fft.rfft2(image_data, norm=self._fft_norm)
        else:
            fft_data_batch = torch.fft.fft2(image_data, norm=self._fft_norm)
        fft_magnitude = torch.abs(fft_data_batch)
        return fft_magnitude

    def get_normalize_transform(self) -> torch.nn.Module:
        return transforms.Normalize((self.norm_mean,), (self.norm_std,))

    def get_inv_normalize_transform(self) -> torch.nn.Module:
        return NormalizeInverse((self.norm_mean,), (self.norm_std,))


def create_data_loaders(ds_name: str, img_size: int,
                        use_aug: bool, use_aug_test: bool, rot_degrees: float,
                        batch_size_train: int, batch_size_test: int,
                        seed: int, use_gan: bool, use_rfft: bool,
                        n_dataloader_workers: int, paired_part: float, fft_norm: str, log: logging.Logger,
                        s3: Optional[S3FileSystem] = None,
                        prob_aug: float = 1.0,
                        gamma_corr: Sequence[float] = (0.85, 1.125),
                        gauss_blur: Sequence[float] = (0.5, 1.5),
                        sharpness_factor: float = 1.0,
                        rnd_vert_flip: bool = False,
                        rnd_horiz_flip: bool = False
                        ):
    log.debug('Create train dataset')
    train_dataset = PhaseRetrievalDataset(ds_name=ds_name, img_size=img_size, train=True,
                                          use_aug=use_aug, rot_degrees=rot_degrees, is_gan=False,
                                          paired_part=paired_part, fft_norm=fft_norm, use_rfft=use_rfft,
                                          log=log, seed=seed, s3=s3, prob_aug=prob_aug,
                                          gamma_corr=gamma_corr,
                                          gauss_blur=gauss_blur,
                                          sharpness_factor=sharpness_factor,
                                          rnd_vert_flip=rnd_vert_flip,
                                          rnd_horiz_flip=rnd_horiz_flip)

    log.debug('Create test dataset')
    test_dataset = PhaseRetrievalDataset(ds_name=ds_name, img_size=img_size, train=False,
                                         use_aug=use_aug_test, rot_degrees=rot_degrees, is_gan=False,
                                         paired_part=1.0, fft_norm=fft_norm, use_rfft=use_rfft,
                                         log=log, seed=seed, s3=s3, prob_aug=prob_aug,
                                         gamma_corr=gamma_corr,
                                         gauss_blur=gauss_blur,
                                         sharpness_factor=sharpness_factor,
                                         rnd_vert_flip=rnd_vert_flip,
                                         rnd_horiz_flip=rnd_horiz_flip
                                         )

    paired_tr_sampler = torch.utils.data.SubsetRandomSampler(train_dataset.paired_ind)
    unpaired_tr_sampler = torch.utils.data.SubsetRandomSampler(train_dataset.unpaired_paired_ind)

    log.debug('Create train  paired loader')
    train_paired_loader = DataLoader(train_dataset, batch_size=batch_size_train,
                                     worker_init_fn=np.random.seed(seed), num_workers=n_dataloader_workers,
                                     sampler=paired_tr_sampler)

    log.debug('Create train  unnpaired loader')
    train_unpaired_loader = DataLoader(train_dataset, batch_size=batch_size_train,
                                       worker_init_fn=np.random.seed(seed), num_workers=n_dataloader_workers,
                                       sampler=unpaired_tr_sampler)

    log.debug('Create test loader')
    test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False,
                             worker_init_fn=np.random.seed(seed),
                             num_workers=n_dataloader_workers)

    return train_paired_loader, train_unpaired_loader, test_loader, train_dataset, test_dataset


def example_mnist_upiared():
    logging.basicConfig(level=logging.DEBUG)
    log = logging.getLogger('exmaple_mnist_upiared')
    train_paired_loader, train_unpaired_loader, test_loader, train_dataset, test_dataset \
        = create_data_loaders(ds_name='mnist',
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
