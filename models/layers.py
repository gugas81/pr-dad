
import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional
from models.untils import get_norm_layer, get_pool_2x2, get_activation


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


class ConvModule(nn.Module):
    def __init__(self,
                 ch_in: int,
                 ch_out: int,
                 kernel_size: int = 3,
                 bias: bool = True,
                 padding: int = 1,
                 img_size: Optional[int] = None,
                 dim_latent: Optional[int] = None,
                 active_type: str = 'leakly_relu',
                 norm_type: str = 'batch_norm'):
        self._conv = nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=1, padding=padding, bias=bias)
        self._norm = get_norm_layer(norm_type=norm_type, n_channel=ch_out, img_size=img_size, dim_latent=dim_latent)
        self._activ = get_activation(active_type)

    def forward(self, x: Tensor, latent_w: Optional[Tensor] = None) -> Tensor:
        x = self._conv(x)
        x = self._norm(x, latent_w)
        x = self._activ(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self,
                 ch_in: int,
                 ch_out: int,
                 ch_inter: Optional[int] = None,
                 kernel_size: int = 3,
                 stride: int = 1,
                 bias: bool = True,
                 padding: int = 1,
                 img_size: Optional[int] = None,
                 dim_latent: Optional[int] = None,
                 active_type: str = 'leakly_relu',
                 norm_type: str = 'batch_norm'):
        super(ConvBlock, self).__init__()
        if ch_inter is None:
            ch_inter = ch_out

        self._conv_sub_block_1 = ConvModule(ch_in=ch_in,
                                            ch_out=ch_inter,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,
                                            bias=bias,
                                            norm_type=norm_type,
                                            n_channel=ch_inter,
                                            img_size=img_size,
                                            dim_latent=dim_latent,
                                            active_type=active_type)

        self._conv_sub_block_2 = ConvModule(ch_in=ch_inter,
                                            ch_out=ch_out,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,
                                            bias=bias,
                                            norm_type=norm_type,
                                            n_channel=ch_inter,
                                            img_size=img_size,
                                            dim_latent=dim_latent,
                                            active_type=active_type)

    def forward(self, x: Tensor, latent_w: Optional[Tensor] = None) -> Tensor:
        x = self._conv_sub_block_1(x, latent_w)
        x = self._conv_sub_block_2(x, latent_w)
        return x


class ResBlock(nn.Module):
    def __init__(self, ch_in: int,
                 ch_out: int,
                 ch_inter: Optional[int] = None,
                 kernel_size: int = 3,
                 stride: int = 1,
                 bias: bool = True,
                 padding: int = 1,
                 img_size: Optional[int] = None,
                 dim_latent: Optional[int] = None,
                 active_type: str = 'leakly_relu',
                 norm_type: str = 'batch_norm'):
        super(ResBlock, self).__init__()
        if ch_inter is None:
            ch_inter = ch_in

        self._conv_sub_block_1 = ConvModule(ch_in=ch_in,
                                            ch_out=ch_inter,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,
                                            bias=bias,
                                            norm_type=norm_type,
                                            n_channel=ch_inter,
                                            img_size=img_size,
                                            dim_latent=dim_latent,
                                            active_type=active_type)

        self._conv_sub_block_2 = ConvModule(ch_in=ch_in,
                                            ch_out=ch_inter,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,
                                            bias=bias,
                                            norm_type=norm_type,
                                            n_channel=ch_inter,
                                            img_size=img_size,
                                            dim_latent=dim_latent,
                                            active_type=active_type)

        ch_inter_out = 2 * ch_inter + ch_out

        self._conv_sub_block_out = ConvModule(ch_in=ch_inter_out,
                                                ch_out=ch_inter,
                                                kernel_size=kernel_size,
                                                stride=stride,
                                                padding=padding,
                                                bias=bias,
                                                norm_type=norm_type,
                                                n_channel=ch_inter,
                                                img_size=img_size,
                                                dim_latent=dim_latent,
                                                active_type=active_type)

    def forward(self, x_in: torch.Tensor, latent_w: Optional[Tensor] = None) -> torch.Tensor:
        x_1 = self._conv_sub_block_1(x_in, latent_w)
        x_2 = self._conv_sub_block_2(x_in, latent_w)
        concat_x = torch.cat((x_in, x_1, x_2), dim=1)
        x_out = self._conv_sub_block_out(concat_x, latent_w)
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
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            get_activation(active_type)
        )

    def forward(self, x):
        x = self.up(x)
        return x



