import torch
from torch import Tensor
import math
import torch.nn as nn


def harmonics(n: int) -> Tensor:
    r"""
    Computes the cosine harmonics for the DCT transform
    """
    spatial = torch.arange(float(n)).reshape((n, 1))
    spectral = torch.arange(float(n)).reshape((1, n))

    spatial = 2 * spatial + 1
    spectral = (spectral * math.pi) / (2 * n)

    return torch.cos(spatial @ spectral)


def normalize_matrix(n: int) -> Tensor:
    r"""
    Computes the constant scale factor which makes the DCT orthonormal
    """
    norm_matrix = torch.ones((n, 1))
    norm_matrix[0, 0] = 1 / math.sqrt(2)
    return norm_matrix @ norm_matrix.t()


class Dct2DForward(nn.Module):
    def __init__(self, img_size: int):
        super(Dct2DForward, self).__init__()
        self._scale_factor = 1 / math.sqrt(2 * img_size)
        self._norm_matrix = nn.Parameter(normalize_matrix(img_size), requires_grad=False)
        self._harmonics_matrix = nn.Parameter(harmonics(img_size), requires_grad=False)

    def forward(self, img: Tensor) -> Tensor:
        coeff = self._scale_factor * self._norm_matrix * (self._harmonics_matrix.t() @ img @ self._harmonics_matrix)
        return coeff


class Dct2DInverse(Dct2DForward):
    def __init__(self, img_size: int):
        super(Dct2DInverse, self).__init__(img_size)

    def forward(self, coeff: Tensor) -> Tensor:
        img = self._scale_factor * (self._norm_matrix @ (self._norm_matrix * coeff) @ self._norm_matrix.t())
        return img





