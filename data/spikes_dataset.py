from typing import Union, Tuple, Dict, Any

import numpy as np
import torch
from scipy.ndimage.filters import gaussian_filter
from torch import Tensor
from torch.utils.data import IterableDataset


def fft_magnitude(img: Union[Tensor, np.ndarray], shift: bool = False, norm: str = "ortho") -> Tensor:
    if isinstance(img, np.ndarray):
        img_tensor = torch.from_numpy(img)
    else:
        img_tensor = img
    fft_img = torch.fft.fft2(img_tensor, norm=norm)
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
    img_spikes = torch.from_numpy(img_spikes).to(dtype=torch.float)
    img_spikes_noised = torch.clone(img_spikes)
    if add_gauss_noise > 0.0:
        rnd_noise = add_gauss_noise * torch.randn_like(img_spikes)
        img_spikes_noised = img_spikes_noised + rnd_noise

    return img_spikes, img_spikes_noised, x, y


class SpikesDataGenerator(IterableDataset):
    def __init__(self,
                 spikes_range: Union[int, Tuple[int, int]] = (3, 16),
                 img_size: int = 32,
                 min_dist: int = 4,
                 sigma: float = 1.0,
                 add_gauss_noise: float = 0.0125,
                 len_ds: int = 10000,
                 shift_fft: bool = False):
        'Initialization'
        self.spikes_range = spikes_range
        self.img_size = img_size
        self.min_dist = min_dist
        self.sigma = sigma
        self.add_gauss_noise = add_gauss_noise
        self.len_ds = len_ds
        self.shift_fft = shift_fft

    def __len__(self):
        'Denotes the total number of samples'
        return int(self.len_ds)

    def _get_item(self):
        'Generates one sample of data'
        # Select sample
        if isinstance(self.spikes_range, int):
            n_spikes = self.spikes_range
        else:
            n_spikes = np.random.randint(self.spikes_range[0], self.spikes_range[1])

        # Load data and get label
        # print(f'n_spikes:{n_spikes}, self.img_size: {self.img_size}')
        img_spikes, img_spikes_noised, x, y = get_random_spike_signals_polar(n_spikes=n_spikes,
                                                                             img_size=self.img_size,
                                                                             min_dist=self.min_dist,
                                                                             sigma=self.sigma,
                                                                             add_gauss_noise=self.add_gauss_noise)
        fft_spkes = fft_magnitude(img_spikes, shift=self.shift_fft)
        fft_spkes_noised = fft_magnitude(img_spikes_noised, shift=self.shift_fft)

        return fft_spkes, fft_spkes_noised, img_spikes, img_spikes_noised, n_spikes, x, y

    def __iter__(self) -> Dict[str, Any]:
        for _ in range(self.len_ds):
            fft_spkes, fft_spkes_noised, img_spikes, img_spikes_noised, n_spikes, x, y = self._get_item()
            x_tensor = np.empty(self.spikes_range[1])
            x_tensor[:] = np.nan
            x_tensor[:x.shape[0]] = x

            y_tensor = np.empty(self.spikes_range[1])
            y_tensor[:] = np.nan
            y_tensor[:x.shape[0]] = y

            item_data = {'fft_spikes': fft_spkes,
                         'fft_spikes_noised': fft_spkes_noised,
                         'img_spikes': img_spikes,
                         'img_spikes_noised': img_spikes_noised,
                         'n_spikes': torch.tensor([n_spikes], dtype=torch.int),
                         'x': torch.tensor(x_tensor, dtype=torch.int),
                         'y': torch.tensor(y_tensor, dtype=torch.int)}
            yield item_data
