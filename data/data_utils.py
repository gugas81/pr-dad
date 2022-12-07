import logging
from dataclasses import dataclass
from typing import Optional

import fire
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from common import ConfigTrainer, S3FileSystem
from data.image_dataset import PhaseRetrievalDataset


@dataclass
class DataHolder:
    train_paired_loader: Optional[DataLoader] = None
    train_unpaired_loader: Optional[DataLoader] = None
    test_loader: Optional[DataLoader] = None
    train_ds: Optional[Dataset] = None
    test_ds: Optional[Dataset] = None


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
