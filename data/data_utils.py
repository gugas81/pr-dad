import logging
from dataclasses import dataclass
from typing import Optional

import fire
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from common import ConfigTrainer, S3FileSystem, ConfigSpikesTrainer
from data.image_dataset import PhaseRetrievalDataset
from data.spikes_dataset import SpikesDataGenerator


@dataclass
class DataHolder:
    train_paired_loader: Optional[DataLoader] = None
    train_unpaired_loader: Optional[DataLoader] = None
    test_loader: Optional[DataLoader] = None
    train_ds: Optional[Dataset] = None
    test_ds: Optional[Dataset] = None


def create_spikes_data_loaders(config: ConfigSpikesTrainer,
                               log: logging.Logger,
                               s3: Optional[S3FileSystem] = None,
                               inv_norm=None) -> DataHolder:
    log.debug('Create spikes dataset and its loaders')
    len_ds_tr = config.n_iter_tr * config.batch_size_train
    spike_generator_tr = SpikesDataGenerator(spikes_range=config.spikes_range,
                                             img_size=config.image_size,
                                             add_gauss_noise=config.gauss_noise,
                                             sigma=config.sigma,
                                             len_ds=len_ds_tr,
                                             pad=config.pad,
                                             shift_fft=config.shift_fft,
                                             log=log,
                                             inv_norm=inv_norm)

    len_ds_ts = config.n_iter_eval * config.batch_size_test
    spike_generator_ts = SpikesDataGenerator(spikes_range=config.spikes_range,
                                             img_size=config.image_size,
                                             add_gauss_noise=config.gauss_noise,
                                             sigma=config.sigma,
                                             len_ds=len_ds_ts,
                                             pad=config.pad,
                                             shift_fft=config.shift_fft,
                                             log=log,
                                             inv_norm=inv_norm)

    spikes_loader_train = DataLoader(spike_generator_tr,
                                     batch_size=config.batch_size_train,
                                     worker_init_fn=np.random.seed(config.seed),
                                     num_workers=config.n_dataloader_workers)
    log.info(f'loader_train_data = {len(spikes_loader_train)} with batch = {config.batch_size_train}')

    spikes_loader_val = DataLoader(spike_generator_ts,
                                   batch_size=config.batch_size_test,
                                   worker_init_fn=np.random.seed(config.seed),
                                   num_workers=config.n_dataloader_workers)
    log.info(f'loader_val_data = {len(spikes_loader_val)} with batch = {config.batch_size_test}')

    return DataHolder(train_paired_loader=spikes_loader_train,
                      test_ds=spike_generator_ts,
                      test_loader=spikes_loader_val,
                      train_ds=spike_generator_tr)


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


def example_mnist_unpaired():
    logging.basicConfig(level=logging.DEBUG)
    log = logging.getLogger('example_mnist_unpaired')
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
