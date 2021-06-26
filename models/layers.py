
import torch
import numpy as np
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FcBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_dropout: bool = False, use_bn: bool = False):
        super(FcBlock, self).__init__()
        self.fc_seq = [nn.Linear(in_features, out_features)]
        if use_bn:
            self.fc_seq.append(nn.BatchNorm1d(out_features))
        self.fc_seq.append(nn.LeakyReLU())
        if use_dropout:
            self.fc_seq.append(nn.Dropout())

        self.fc_seq = nn.Sequential(*self.fc_seq)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc_seq(x)


class ConvBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, ch_inter: Optional[int] = None):
        super(ConvBlock, self).__init__()
        if ch_inter is None:
            ch_inter = ch_out
        self.conv_block = nn.Sequential(
            nn.Conv2d(ch_in, ch_inter, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_inter),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_inter, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_block(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1, bias: bool = False,
                 bias_out: bool = True):
        super(ResBlock, self).__init__()

        self._conv_block1 = nn.Sequential(nn.BatchNorm2d(in_channels, affine=True),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, bias=bias))

        self._conv_block2 = nn.Sequential(nn.BatchNorm2d(in_channels, affine=True),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, bias=bias))

        self._conv_block_out = nn.Sequential(nn.BatchNorm2d(3*in_channels, affine=True),
                                             nn.ReLU(inplace=True),
                                             nn.Conv2d(3*in_channels, out_channels, kernel_size,
                                                       padding=padding, bias=bias_out))


    # pylint: disable=arguments-differ
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_1 = self._conv_block1(x)
        x_2 = self._conv_block1(x)
        concat_x = torch.cat((x, x_1, x_2), dim=1)
        x_out = self._conv_block_out(concat_x)
        return x_out


class DownConvBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super(DownConvBlock, self).__init__()
        self.conv_block = nn.Sequential(ConvBlock(ch_in, ch_out), nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_block(x)
        return x


class UpConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpConvBlock, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x



