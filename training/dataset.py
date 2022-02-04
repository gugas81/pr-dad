import logging
import os
from functools import partial
from typing import Union, Tuple, Dict, Any, Optional, Callable, Sequence, List
import fire
from tqdm import tqdm
import PIL
import numpy as np
import torch
import torchvision
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from common import ConfigTrainer
import common.utils as utils
from training.augmentations import RandomGammaCorrection
from common import PATHS, S3FileSystem, NormalizeInverse, DataBatch
from models.torch_dct import Dct2DForward, Dct2DInverse


class PhaseRetrievalDataset(Dataset):
    def __init__(self,
                 config: ConfigTrainer,
                 is_train: bool,
                 log: logging.Logger,
                 s3: Optional[S3FileSystem] = None,
                 is_gan: bool = False
                 ):
        def celeba_ds(root: str, train: bool, download: bool, transform: Optional[Callable] = None) ->torchvision.datasets.CelebA:
            return torchvision.datasets.CelebA(root=root,
                                               split='train' if train else 'test',
                                               download=download,
                                               transform=transform)
            # return CelebASmallGray(root=root,
            #                        split='train' if train else 'test',
            #                        download=download,
            #                        transform=transform,
            #                        image_size=self._config.image_size)

        np.random.seed(seed=config.seed)
        self._config = config
        self._fft_norm = config.fft_norm
        self._is_train = is_train
        self._is_gan = is_gan
        self._use_aug = config.use_aug if self._is_train else config.use_aug_test
        self._use_rfft = config.use_rfft
        self._log = log
        self._s3 = S3FileSystem() if s3 is None else s3

        if self._config.use_dct_input:
            self.dct_input = Dct2DForward(utils.get_padded_size(self._config.image_size, self._config.add_pad))
        else:
            self.dct_input = None

        ds_name = config.dataset_name.lower()
        ds_path = os.path.join(PATHS.DATASETS, ds_name)
        alignment_transform = transforms.Resize(self._config.image_size)
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
                transforms.CenterCrop(self._config.image_crop_size),
                transforms.Resize(self._config.image_size)
                ])
            normalize_transform = transforms.Normalize((self.norm_mean, self.norm_mean, self.norm_mean),
                                                       (self.norm_std, self.norm_std, self.norm_std))
        elif ds_name == 'cifar10':
            ds_class = torchvision.datasets.CIFAR10
            is_rgb = True
            self.norm_mean = 0.48
            self.norm_std = 0.20
            normalize_transform = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        else:
            raise NameError(f'Not valid ds type {ds_name}')

        data_transforms = [alignment_transform, transforms.ToTensor(), normalize_transform]
        if is_rgb:
            data_transforms.append(transforms.Grayscale(num_output_channels=1))

        # if ds_name == 'celeba':
        #     data_transforms = []

        if self._use_aug:
            prob_aug = self._config.prob_aug
            augmentations_transforms = []
            if self._config.sharpness_factor is not None:
                augmentations_transforms.append(transforms.RandomAdjustSharpness(sharpness_factor=1.0, p=prob_aug))

            if self._config.gauss_blur is not None:
                augmentations_transforms.append(transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3, sigma=self._config.gauss_blur)], p=prob_aug))

            if self._config.gamma_corr is not None:
                augmentations_transforms.append(transforms.RandomApply([
                    RandomGammaCorrection(gamma_range=self._config.gamma_corr)], p=prob_aug))

            if len(augmentations_transforms) > 0:
                augmentations_transforms = [transforms.RandomOrder(augmentations_transforms)]

            if self._config.rnd_vert_flip:
                augmentations_transforms.append(transforms.RandomVerticalFlip(p=prob_aug))

            if self._config.rnd_horiz_flip:
                augmentations_transforms.append(transforms.RandomHorizontalFlip(p=prob_aug))

            if self._config.affine_aug:
                pad_val = int(0.25 * self._config.image_size *
                              (self._config.scale[1] if (self._config.scale is not None) else 1))
                aff_tran = transforms.RandomApply([
                    transforms.Pad(pad_val, padding_mode='reflect'),
                    transforms.RandomAffine(degrees=self._config.rot_degrees,
                                            translate=self._config.translation,
                                            scale=self._config.scale,
                                            interpolation=InterpolationMode.BILINEAR),
                    transforms.CenterCrop(self._config.image_size)
                ], p=prob_aug)
                augmentations_transforms = [aff_tran] + augmentations_transforms

            if isinstance(augmentations_transforms, list) and len(augmentations_transforms) > 0:
                augmentations_transforms = transforms.Compose(augmentations_transforms)
                data_transforms.append(augmentations_transforms)

        if isinstance(data_transforms, list) and len(data_transforms) > 0:
            data_transforms = transforms.Compose(data_transforms)

        if isinstance(data_transforms, list) and len(data_transforms) == 0:
            data_transforms = None

        self.image_dataset = ds_class(root=ds_path,
                                      train=self._is_train,
                                      download=True,
                                      transform=data_transforms)
        self.len_ds = len(self.image_dataset)
        if self._config.part_supervised_pairs < 1.0:
            paired_len = int(self._config.part_supervised_pairs * self.len_ds)
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
        def label_to_tensor(x) -> Tensor:
            if torch.is_tensor(x):
                return x
            else:
                return torch.from_numpy(np.array(x))
        is_paired = idx in self.paired_ind
        img_item = self.image_dataset[idx]
        image_data = img_item[0].to(device='cpu')
        label = label_to_tensor(img_item[1]).to(device='cpu')
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
        if self._config.add_pad > 0.0:
            pad_value = utils.get_pad_val(self._config.image_size, self._config.add_pad)
            image_data_pad = transforms.functional.pad(image_data, pad_value, padding_mode='edge')
        else:
            image_data_pad = image_data
        if self._config.use_dct_input:
            fft_data_batch = self.dct_input(image_data_pad)
        elif self._use_rfft:
            fft_data_batch = torch.fft.rfft2(image_data_pad, norm=self._fft_norm)
        else:
            fft_data_batch = torch.fft.fft2(image_data_pad, norm=self._fft_norm)
        fft_magnitude = self._config.spectral_factor * torch.abs(fft_data_batch)
        return fft_magnitude

    def get_normalize_transform(self) -> torch.nn.Module:
        return transforms.Normalize((self.norm_mean,), (self.norm_std,))

    def get_inv_normalize_transform(self) -> torch.nn.Module:
        return NormalizeInverse((self.norm_mean,), (self.norm_std,))


class CelebASmallGray(torchvision.datasets.CelebA):
    def __init__(
            self,
            root: str,
            split: str = "train",
            target_type: Union[List[str], str] = "attr",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            image_size: int = 32
    ) -> None:
        super(CelebASmallGray, self).__init__(root,
                                              transform=transform,
                                               target_transform=target_transform,
                                              split=split,
                                              target_type=target_type,
                                              download=download)
        image_path = os.path.join(self.root, self.base_folder, f"img_align_celeba_{image_size}")
        data_file_file = os.path.join(image_path, f'img_data_{split}.pt')
        assert os.path.isfile(data_file_file)
        self._img_data = torch.load(data_file_file, map_location=torch.device('cpu')).detach().numpy()

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # img_out_path = os.path.join(self.image_path, f'{self.filename[index]}'.pt)
        # X = torch.load(img_out_path)
        # X = PIL.Image.open(os.path.join(self.image_path, self.filename[index]))
        X = torch.tensor(self._img_data[index], device=torch.device('cpu'))

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target


def create_down_sampled_celeba(image_size: int = 32):
    def get_celeba_ds(root: str, split_: str, download: bool, transform: Optional[Callable] = None) -> torchvision.datasets.CelebA:
        return torchvision.datasets.CelebA(root=root,
                                           split=split_,
                                           download=download,
                                           transform=transform)

    alignment_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size)])

    norm_mean = 0.5
    norm_std = 0.5

    normalize_transform = transforms.Normalize((norm_mean, norm_mean, norm_mean),
                                               (norm_std, norm_std, norm_std))

    data_transforms = [alignment_transform, transforms.ToTensor(), normalize_transform, transforms.Grayscale(num_output_channels=1)]

    # data_transforms = [alignment_transform, transforms.Grayscale(num_output_channels=1), transforms.ToTensor()]
    data_transforms = transforms.Compose(data_transforms)

    splits = ['train', 'test']
    for split in splits:
        ds_path = os.path.join(PATHS.DATASETS, 'celeba')
        celeba_ds = get_celeba_ds(root=ds_path, split_=split, download=True,  transform=[])
        out_img_path = os.path.join(celeba_ds.root, celeba_ds.base_folder, f"img_align_celeba_{image_size}")
        os.makedirs(out_img_path, exist_ok=True)
        len_ds = len(celeba_ds)
        img_data = [celeba_ds[ds_ind][0] for ds_ind in tqdm(range(len_ds))]
        img_data: Tensor = torch.stack(img_data)
        out_file = os.path.join(out_img_path, f'img_data_{split}.pt')
        print(f'Image data: {img_data.shape} will saved in: {out_file}')
        torch.save(img_data, out_file)
        # for ds_ind in tqdm(range(len_ds)):
        #     img, _ = celeba_ds[ds_ind]
        #     img_data.append(img)
            # img_out_path = os.path.join(out_img_path, f'{celeba_ds.filename[ds_ind]}'.pt)
            # torch.save(img, img_out_path)
            # img.save(img_out_path)


def create_data_loaders(config: ConfigTrainer,
                        log: logging.Logger,
                        s3: Optional[S3FileSystem] = None):
    log.debug('Create train dataset')
    train_dataset = PhaseRetrievalDataset(config=config,
                                          is_train=True,
                                          is_gan=False,
                                          log=log,
                                          s3=s3)

    log.debug('Create test dataset')
    test_dataset = PhaseRetrievalDataset(config=config,
                                         is_train=False,
                                         is_gan=False,
                                         log=log,
                                         s3=s3)

    paired_tr_sampler = torch.utils.data.SubsetRandomSampler(train_dataset.paired_ind)
    unpaired_tr_sampler = torch.utils.data.SubsetRandomSampler(train_dataset.unpaired_paired_ind)

    log.debug('Create train  paired loader')
    train_paired_loader = DataLoader(train_dataset,
                                     batch_size=config.batch_size_train,
                                     worker_init_fn=np.random.seed(config.seed),
                                     num_workers=config.n_dataloader_workers,
                                     sampler=paired_tr_sampler)

    log.debug('Create train  unnpaired loader')
    train_unpaired_loader = DataLoader(train_dataset,
                                       batch_size=config.batch_size_train,
                                       worker_init_fn=np.random.seed(config.seed),
                                       num_workers=config.n_dataloader_workers,
                                       sampler=unpaired_tr_sampler)

    log.debug('Create test loader')
    test_loader = DataLoader(test_dataset,
                             batch_size=config.batch_size_test,
                             shuffle=False,
                             worker_init_fn=np.random.seed(config.seed),
                             num_workers=config.n_dataloader_workers)

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
