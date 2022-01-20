import torch
import torch.nn as nn
from torch import Tensor


class BaseAe(nn.Module):
    def __init__(self, img_size: int, in_ch: int = 1, deep: int = 3):
        super(BaseAe, self).__init__()
        self._img_size = img_size
        self._deep = deep
        self._in_ch = in_ch

    @property
    def n_enc_features_ch(self) -> int:
        raise NotImplementedError

    @property
    def n_dec_features_ch(self) -> int:
        raise NotImplementedError

    @property
    def n_features_size(self) -> int:
        raise NotImplementedError

    def encode(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def decode(self, features: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, x: Tensor) -> (Tensor, Tensor, Tensor, Tensor):
        raise NotImplementedError

    def map_to_dec_features(self, enc_features: Tensor) -> (Tensor, Tensor):
        raise NotImplementedError

    def extra_repr(self) -> str:
        return f'img_size={self._img_size}, ' \
               f'deep={self._deep}, ' \
               f'img_channels={self._in_ch},  ' \

