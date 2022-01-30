import numpy as np
import random
import torch
from torch import Tensor
from torch import nn
from typing import List, Optional
import torchvision


def set_seed(seed: int):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def hardshrink_subbands(subbands: Tensor, lambd: float = 0.5) -> Tensor:
    subbands[:, 1:] = nn.Hardshrink(lambd=lambd)(subbands[:, 1:])
    return subbands


def softdshrink_subbands(subbands: Tensor, lambd: float = 0.5) -> Tensor:
    subbands[:, 1:] = nn.Softshrink(lambd=lambd)(subbands[:, 1:])
    return subbands


class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return super().__call__(tensor.clone())


def get_fft2_freq(img_size: int, use_rfft: bool) -> List[int]:
    if use_rfft:
        fft_freq_1d = torch.fft.rfftfreq(img_size)
    else:
        fft_freq_1d = torch.fft.fftfreq(img_size)
    return (fft_freq_1d * img_size).numpy().astype(int).tolist()


def get_pad_val(img_size: int, add_pad: float) -> int:
    return int(0.5 * add_pad * img_size)


def get_padded_size(img_size: int, add_pad: float) -> int:
    pad_val = get_pad_val(img_size, add_pad)
    return 2 * pad_val + img_size


def get_fft2_last_size(img_size: int, use_rfft: bool) -> int:
    return len(get_fft2_freq(img_size, use_rfft))


def get_magnitude_size_2d(img_size: int, pad_val: int, use_rfft: bool) -> List[int]:
    pad_size = get_padded_size(img_size, pad_val)
    return [pad_size, get_fft2_last_size(pad_size, use_rfft)]


def get_flatten_fft2_size(img_size:  int, use_rfft: bool) -> int:
    return img_size * get_fft2_last_size(img_size, use_rfft)

