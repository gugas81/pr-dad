
import torch
import numpy as np
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from models.untils import get_norm_layer, get_pool_2x2, get_activation, BlockList


class FcBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_dropout: bool = False, norm_type: str = None,
                 active_type: str = 'leakly_relu', active_params: int = 1):
        super(FcBlock, self).__init__()
        self.fc_seq = BlockList()
        self.fc_seq.append(nn.Linear(in_features, out_features))
        self.fc_seq.append(get_norm_layer(name_type=norm_type, input_ch=out_features, is_2d=False))
        self.fc_seq.append(get_activation(active_type, active_params))
        if use_dropout:
            self.fc_seq.append(nn.Dropout())

    def forward(self, x: Tensor) -> Tensor:
        return self.fc_seq(x)


class ConvBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, ch_inter: Optional[int] = None, active_type: str = 'leakly_relu',
                 padding_mode: str = 'zeros'):
        super(ConvBlock, self).__init__()
        if ch_inter is None:
            ch_inter = ch_out
        self.conv_block = nn.Sequential(
            nn.Conv2d(ch_in, ch_inter, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode),
            nn.BatchNorm2d(ch_inter),
            get_activation(active_type),
            nn.Conv2d(ch_inter, ch_out, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode),
            nn.BatchNorm2d(ch_out),
            get_activation(active_type)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_block(x)
        return x


class SpatialAttentionBlock(nn.Module):
    def __init__(self, in_ch: int,  kernel_size: int = 3):
        super(SpatialAttentionBlock, self).__init__()

        self.conv_s = nn.Conv2d(in_ch, in_ch, kernel_size, padding=kernel_size//2, bias=False)

    @staticmethod
    def soft_max_special(x):
        x_flatten = x.view(x.size(0), x.size(1), -1)
        return nn.Softmax(dim=2)(x_flatten)

    def forward(self, features_input: Tensor) -> Tensor:
        s_att = self.conv_s(features_input)
        s_att = self.soft_max_special(s_att)
        s_max, _ = torch.max(s_att.abs(), dim=2, keepdim=True)
        s_att /= s_max
        return s_att.view_as(features_input)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1, bias: bool = False,
                 bias_out: bool = True, active_type: str = 'leakly_relu'):
        super(ResBlock, self).__init__()

        self._conv_block1 = nn.Sequential(nn.BatchNorm2d(in_channels, affine=True),
                                          get_activation(active_type),
                                          nn.Conv2d(in_channels, in_channels, kernel_size,
                                                    padding=padding, bias=bias, padding_mode='replicate'))

        self._conv_block2 = nn.Sequential(nn.BatchNorm2d(in_channels, affine=True),
                                          get_activation(active_type),
                                          nn.Conv2d(in_channels, in_channels, kernel_size,
                                                    padding=padding, bias=bias, padding_mode='replicate'))

        self._conv_block_out = nn.Sequential(nn.BatchNorm2d(3*in_channels, affine=True),
                                             get_activation(active_type),
                                             nn.Conv2d(3*in_channels, out_channels, kernel_size,
                                                       padding=padding, bias=bias_out, padding_mode='replicate'))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_1 = self._conv_block1(x)
        x_2 = self._conv_block1(x)
        concat_x = torch.cat((x, x_1, x_2), dim=1)
        x_out = self._conv_block_out(concat_x)
        return x_out


class DownConvBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, use_res_block:  bool = False,
                 down_pool: str = 'avrg_pool', active_type: str = 'leakly_relu'):
        super(DownConvBlock, self).__init__()
        if use_res_block:
            conv_op = ResBlock
        else:
            conv_op = ConvBlock
        self.conv_block = nn.Sequential(get_pool_2x2(down_pool), conv_op(ch_in, ch_out, active_type=active_type))

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_block(x)
        return x


class UpConvBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, up_mode: str = 'nearest', active_type: str = 'leakly_relu'):
        # ``'nearest'``,
        # ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``.
        # Default: ``'nearest'``
        super(UpConvBlock, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=up_mode, align_corners=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1,padding_mode='replicate'),
            nn.BatchNorm2d(ch_out),
            get_activation(active_type)
        )

    def forward(self, x):
        x = self.up(x)
        return x



