import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import random
from torchvision import transforms
import math
import matplotlib.pyplot as plt
from scipy.fft import fft, rfft, irfft, ifft,fftshift, fft2, ifft2, dct, idct
from scipy.ndimage.filters import gaussian_filter
from PIL import Image, ImageCms
from torch.utils.data.sampler import Sampler
from models.seq_blocks import MlpNet


def fft_magitude(img, shift: bool=True):
    if isinstance(img, np.ndarray):
        img_tensor = torch.from_numpy(img)
    else:
        img_tensor = img
    fft_img = torch.fft.fft2(img_tensor, norm="forward")
    fft_img_magnitude = torch.abs(fft_img)
    if shift:
        fft_img_magnitude = torch.fft.fftshift(fft_img_magnitude)
    return fft_img_magnitude

def pol2cart(r, phi):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y


def get_random_spike_signals_polar(n_spikes: int = 1,
                                   img_size: int = 32,
                                   min_dist: int = 4,
                                   sigma: float = 1.0,
                                   add_gauss_noise: float = 0.0125) -> torch.Tensor:
    phi_ranges = 2 * np.pi * np.arange(0, n_spikes, 1) / n_spikes
    half_size = img_size // 2
    r_ranges = (np.random.permutation(half_size - 2 * min_dist) + min_dist)[:n_spikes]
    assert len(r_ranges) == n_spikes
    x, y = pol2cart(r_ranges, phi_ranges)
    x += half_size
    y += half_size
    x = np.int32(np.round(x))
    y = np.int32(np.round(y))
    img_spikes = np.zeros((img_size, img_size))
    for ind in range(n_spikes):
        img_spikes[y[ind], x[ind]] = 1.0
    if sigma > 0:
        img_spikes = gaussian_filter(img_spikes, sigma)
    img_spikes = torch.from_numpy(img_spikes)
    if add_gauss_noise > 0.0:
        img_spikes = img_spikes + add_gauss_noise * torch.randn_like(img_spikes)

    return img_spikes, x, y


class SpikesSampler(Sampler):
    def __init__(self,
                 batch_size: int = 4,
                 spikes_range=(3, 16),
                 img_size: int = 32,
                 min_dist: int = 4,
                 sigma: float = 1.0,
                 add_gauss_noise: float = 0.0125,
                 len_ds: int = 10000):
        'Initialization'
        self.batch_size = batch_size
        self.spikes_range = spikes_range
        self.img_size = img_size
        self.min_dist = min_dist
        self.sigma = sigma
        self.add_gauss_noise = add_gauss_noise
        self.len_ds = len_ds

    def __len__(self):
        'Denotes the total number of samples'
        raise int(self.len_ds)

    def _get_item(self):
        'Generates one sample of data'
        # Select sample
        n_spikes = np.random.randint(self.spikes_range[0], self.spikes_range[1])

        # Load data and get label
        # print(f'n_spikes:{n_spikes}, self.img_size: {self.img_size}')
        img_spikes, x, y = get_random_spike_signals_polar(n_spikes=n_spikes,
                                                          img_size=self.img_size,
                                                          min_dist=self.min_dist,
                                                          sigma=self.sigma,
                                                          add_gauss_noise=self.add_gauss_noise)
        fft_spkes = fft_magitude(img_spikes)

        return fft_spkes, img_spikes, n_spikes, x, y

    def __iter__(self):
        batch = {'fft_spikes': [], 'img_spikes': [], 'n_spikes': [], 'x': [], 'y': []}
        for _ in range(self.batch_size):
            fft_spkes, img_spikes, n_spikes, x, y = self._get_item()
            batch['fft_spikes'].append(fft_spkes)
            batch['img_spikes'].append(img_spikes)
            batch['n_spikes'].append(n_spikes)
            batch['x'].append(x)
            batch['y'].append(y)
        batch['fft_spikes'] = torch.stack(batch['fft_spikes'])
        batch['n_spikes'] = torch.tensor(batch['n_spikes'])[:, None]
        yield batch


if __name__ == '__main__':
    img_size = 32
    iters = 10000
    spike_sampler = SpikesSampler(spikes_range=(2, 8), img_size=img_size, batch_size=16)

    in_size = img_size**2
    ch_list = [in_size, in_size // 4, in_size // 16, in_size // 32, in_size // 128, in_size // 256, 1]
    mlp_net = MlpNet(in_ch=in_size,
                     ch_list = ch_list,
                      out_ch=1,
                     deep=len(ch_list),
                      multy_coeff=0.5)

    mlp_net.train()
    l2_loss = torch.nn.MSELoss()
    optim = optim.Adam(params=mlp_net.parameters(), lr= 0.00001)

    for ind in range(iters):
        batch_spikes_data = next(iter(spike_sampler))
        x = torch.flatten(batch_spikes_data['fft_spikes'], 1).to(torch.float)
        y = batch_spikes_data['n_spikes'].to(torch.float)
        y_pred = mlp_net(x)
        pred_loss = l2_loss(y, y_pred)
        pred_loss.backward()
        optim.step()
        if ind % 100 == 0:
            print(f'ind: {ind}, loss: {pred_loss}')


