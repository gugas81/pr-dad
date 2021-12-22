import torch
import torch.nn as nn
import math
import os
import pickle
import torch
from torch.autograd import Variable
import torchvision
from torch import Tensor
import numpy as np
from typing import Tuple
from PIL import Image
import matplotlib.pyplot as plt
import wget
from torchvision import transforms
import numbers
import random
import math
import torch.nn as nn

from common import PATHS

WAVELET_HAAR_WEIGHTS_PATH = os.path.join(PATHS.PROJECT_ROOT, 'models', 'wavelet_haar_weights_c2.pkl')


class WaveletHaarTransform(nn.Module):
    def __init__(self, img_channels: int = 1, scale: int = 1, decomposition: bool = True,  transpose=True,
                 params_path: str = WAVELET_HAAR_WEIGHTS_PATH):
        super(WaveletHaarTransform, self).__init__()
        assert img_channels == 1 or img_channels == 3
        self.scale = scale
        self.decomposition = decomposition
        self.transpose = transpose
        self.img_channels = img_channels

        self.kernel_size = int(math.pow(2, self.scale))
        self.subband_channels = img_channels * self.kernel_size * self.kernel_size

        if self.decomposition:
            self.wavelet_conv_transform = nn.Conv2d(in_channels=img_channels,
                                                    out_channels=self.subband_channels,
                                                    kernel_size=self.kernel_size,
                                                    stride=self.kernel_size,
                                                    padding=0,
                                                    groups=img_channels,
                                                    bias=False)
        else:
            self.wavelet_conv_transform = nn.ConvTranspose2d(in_channels=self.subband_channels,
                                                             out_channels=img_channels,
                                                             kernel_size=self.kernel_size,
                                                             stride=self.kernel_size,
                                                             padding=0,
                                                             groups=img_channels,
                                                             bias=False)

        self._init_filters(params_path)

    def extra_repr(self) -> str:
        return f'decomposition={self.decomposition}, ' \
               f'transpose={self.transpose}' \
               f'scale={self.scale}, ' \
               f'img_channels={self.img_channels},  ' \
               f'subband_channels={self.subband_channels}, ' \
               f'kernel_size={self.kernel_size}'

    def _init_filters(self, params_path: str):
        assert os.path.isfile(params_path)
        with open(params_path, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            conv_wavelets_weights = u.load()
        conv_filter = torch.from_numpy(conv_wavelets_weights['rec%d' % self.kernel_size])[:self.subband_channels]
        for submodules in self.modules():
            if isinstance(submodules, nn.Conv2d) or isinstance(submodules, nn.ConvTranspose2d):
                submodules.weight.data = conv_filter
                submodules.weight.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        if self.decomposition:
            output = self.wavelet_conv_transform(x)
            if self.transpose:
                ous_size = output.size()
                output = output.view(ous_size[0], self.img_channels, -1, ous_size[2], ous_size[3]).transpose(1, 2).contiguous().view(
                    ous_size)
        else:
            if self.transpose:
                xx = x
                in_size = xx.size()
                xx = xx.view(in_size[0], -1, self.img_channels, in_size[2], in_size[3]).transpose(1, 2).contiguous().view(in_size)
            output = self.wavelet_conv_transform(xx)
        return output


class WaveletHaarTransformAutoencoder(nn.Module):
    def __init__(self, in_ch: int = 1, deep: int = 3):
        super(WaveletHaarTransformAutoencoder, self).__init__()
        self._encoder = WaveletHaarTransform(img_channels=in_ch, scale=deep, decomposition=True)
        self._decoder = WaveletHaarTransform(img_channels=in_ch, scale=deep, decomposition=False)

    def encode(self, x: Tensor) -> Tensor:
        features = self._encoder(x)
        return features

    def forward(self, features: Tensor) -> Tensor:
        x_out = self._decoder(features)
        return x_out

    def forward(self, x: Tensor) -> (Tensor, Tensor, Tensor, Tensor):
        enc_features = self.encode(x)
        dec_features = enc_features
        coeff = enc_features
        x_out = self.decode(dec_features)

        return x_out, enc_features, dec_features, coeff
