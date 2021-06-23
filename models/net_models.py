import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from typing import List, Optional
from enum import Enum
from models.layers import FcBlock, ConvBlock, UpConvBlock, get_norm_layer, DownConvBlock, ResBlock
from common.data_classes import DiscriminatorBatch


class PhaseRetrievalPredictor(nn.Module):
    # class TypeRecon(Enum):

    def __init__(self, use_dropout: bool = False, im_img_size: int = 28, inter_ch: int = 1,  out_ch: int = 1,
                 out_img_size: int = 32,
                 multy_coeff: int = 1, use_bn: bool = False, fft_norm: str = "ortho", deep_fc: int = 4,
                 deep_conv: int = 2,
                 type_recon: str = 'spectral', conv_type: str = 'ConvBlock'):
        super(PhaseRetrievalPredictor, self).__init__()
        self.im_img_size = im_img_size
        self.out_img_size = out_img_size
        self._type_recon = type_recon

        self.out_ch = out_ch
        self.int_ch = inter_ch
        self.multy_coeff = multy_coeff
        self.in_features = self.im_img_size ** 2
        self.inter_features = self.int_ch * self.out_img_size ** 2

        self.out_features = self.out_ch * self.out_img_size ** 2
        self._fft_norm = fft_norm

        if type_recon == 'spectral':
            out_fc_features = 2 * self.inter_features
        elif type_recon == 'phase':
            out_fc_features = 2 * self.inter_features

        out_fc = self.in_features
        in_fc = self.in_features
        self.fc_blocks = []
        for ind in range(deep_fc):
            fc_block = FcBlock(in_fc, out_fc, use_dropout=use_dropout, use_bn=use_bn)
            in_fc = out_fc

            out_fc *= self.multy_coeff
            if ind == deep_fc - 1:
                out_fc = out_fc_features
            self.fc_blocks.append(fc_block)

        # fc1 = FcBlock(self.in_features, self.in_features, use_dropout=use_dropout, use_bn=use_bn)
        # fc2 = FcBlock(self.in_features, self.in_features * self.multy_coeff, use_dropout=use_dropout, use_bn=use_bn)
        # fc3 = FcBlock(self.in_features * self.multy_coeff, self.in_features * (self.multy_coeff ** 2),
        #               use_dropout=use_dropout, use_bn=use_bn)
        # fc4 = FcBlock(self.in_features * (self.multy_coeff ** 2), out_fc_features,
        #               use_dropout=use_dropout, use_bn=use_bn)

        self.fc_blocks = nn.Sequential(*self.fc_blocks)

        if conv_type == 'ConvBlock':
            conv_block_class = ConvBlock
        elif conv_type == 'ResBlock':
            conv_block_class = ResBlock
        else:
            raise NameError(f'Non valid conv_type: {conv_type}')

        in_conv = self.int_ch
        out_conv = 2 * in_conv
        self.conv_blocks = []
        for ind in range(deep_conv):
            conv_block = conv_block_class(in_conv, out_conv)
            in_conv = out_conv
            self.conv_blocks.append(conv_block)

        # conv_block1 = ConvBlock(self.out_ch, 2*self.out_ch)
        # conv_block1 = ResBlock(self.int_ch, 2 * self.int_ch)
        # conv_block2 = ConvBlock(2*self.out_ch, 2 * self.out_ch)
        # conv_block2 = ResBlock(2 * self.int_ch, 2 * self.int_ch)
        conv_out = nn.Conv2d(out_conv, self.out_ch, kernel_size=1, stride=1, padding=0)
        self.conv_blocks.append(conv_out)
        self.conv_blocks = nn.Sequential(*self.conv_blocks)

    def forward(self, magnitude: Tensor) -> (Tensor, Tensor):
        # x = torch.fft.fftshift(x, dim=(-2, -1))
        x_float = magnitude.view(-1, self.in_features)
        fc_features = self.fc_blocks(x_float)

        if self._type_recon == 'spectral':
            spectral = fc_features.view(-1, self.int_ch, self.out_img_size, self.out_img_size, 2)
            spectral = torch.view_as_complex(spectral)
        elif self._type_recon == 'phase':
            phase = fc_features.view(-1, self.int_ch, self.out_img_size, self.out_img_size)
            exp_phase = torch.exp(torch.view_as_complex(torch.stack([torch.zeros_like(magnitude), phase], -1)))
            spectral = magnitude * exp_phase
        intermediate_features = torch.fft.ifft2(spectral, norm=self._fft_norm)

        out_features = self.conv_blocks(intermediate_features.real)
        return out_features, intermediate_features


class UNetConv(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(UNetConv, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_1 = ConvBlock(ch_in=img_ch, ch_out=16)
        self.conv_2 = ConvBlock(ch_in=16, ch_out=32)
        self.conv_3 = ConvBlock(ch_in=32, ch_out=64)

        self.up_3 = UpConvBlock(ch_in=64, ch_out=32)
        self.up_conv_3 = ConvBlock(ch_in=64, ch_out=32)

        self.up_2 = UpConvBlock(ch_in=32, ch_out=16)
        self.up_conv_2 = ConvBlock(ch_in=32, ch_out=16)

        self.conv_out = nn.Conv2d(16, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        # encoding path
        x1 = self.conv_1(x) # 16x27x27

        x2 = self.maxpool(x1)# 16x14x14
        x2 = self.conv_2(x2) # 32x14x14

        x3 = self.maxpool(x2)# 32x7x7
        x3 = self.conv_3(x3) # 64x7x7

        d3 = self.up_3(x3) # 32x14x14
        d3 = torch.cat((x2, d3), dim=1)  # 32x14x14
        d3 = self.up_conv_3(d3)  # 32x14x14

        d2 = self.up_2(d3) # 16x28x28
        d2 = torch.cat((x1, d2), dim=1)  # 16x28x28
        d2 = self.up_conv_2(d2) # 16x28x28

        d1 = self.conv_out(d2) # 1x28x28
        return d1


class ConvUnetPhaseRetrieval(UNetConv):
    def __init__(self):
        super(ConvUnetPhaseRetrieval, self).__init__(img_ch=1, output_ch=2)

    def forward(self, x: Tensor) -> Tensor:
        b, c,  h, w = x.shape
        x = torch.fft.fftshift(x, dim=(-2, -1))
        x = super(ConvUnetPhaseRetrieval, self).forward(x)
        x = torch.view_as_complex(x.permute(0, 2, 3, 1).reshape(-1, 2))
        x = x.view(b, c, h, w)
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        return x


class ConvUnetPRanglePred(UNetConv):
    def __init__(self):
        super(ConvUnetPRanglePred, self).__init__(img_ch=1, output_ch=1)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.fft.fftshift(x, dim=(-2, -1))
        angle_pred = super(ConvUnetPRanglePred, self).forward(x)
        x = x * torch.exp(torch.view_as_complex(torch.stack([torch.zeros_like(x), angle_pred], -1)))
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        return x


class EncoderConv(nn.Module):
    def __init__(self, in_ch=1, encoder_ch=8, deep: int = 3, last_down: bool = True):
        super(EncoderConv, self).__init__()
        self.encoder_ch = encoder_ch
        self.deep = deep

        self.conv_down_blocks = []
        inp_ch_block = in_ch
        self.out_ch = encoder_ch
        for ind_block in range(self.deep):

            if ind_block == self.deep - 1 and last_down:
                conv_block = ConvBlock(ch_in=inp_ch_block, ch_out=self.out_ch)
            else:
                conv_block = DownConvBlock(ch_in=inp_ch_block, ch_out=self.out_ch)
            inp_ch_block = self.out_ch

            if ind_block < self.deep - 1:
                self.out_ch *= 2

            self.conv_down_blocks.append(conv_block)
        self.conv_down_blocks = nn.Sequential(*self.conv_down_blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv_down_blocks(x)


class DecoderConv(nn.Module):
    def __init__(self, img_ch: int = 32, output_ch: int = 1, deep: int = 3):
        super(DecoderConv, self).__init__()
        # assert deep == 3
        self.deep = deep
        self.conv_blocks = []
        ch_im = img_ch
        for ind_block in range(self.deep):

            if ind_block == self.deep - 1:
                conv_block = nn.Conv2d(ch_out, output_ch, kernel_size=1, stride=1, padding=0)
            else:
                ch_out = ch_im // 2
                up_conv_block = UpConvBlock(ch_in=ch_im, ch_out=ch_out)
                self.conv_blocks.append(up_conv_block)
                conv_block = ConvBlock(ch_out, ch_out)

            ch_im = ch_out
            self.conv_blocks.append(conv_block)
        self.conv_blocks = nn.Sequential(*self.conv_blocks)
        #
        # up_conv1 = UpConvBlock(ch_in=img_ch, ch_out=img_ch // 2)   # 32x14x14
        # conv1 = ConvBlock(ch_in=img_ch // 2, ch_out=img_ch // 2)  # 32x14x14
        #
        # up_conv2 = UpConvBlock(ch_in=img_ch // 2, ch_out=img_ch // 4)  # 16x28x28
        # conv2 = ConvBlock(ch_in=img_ch // 4, ch_out=img_ch // 4)  # 16x28x28
        #
        # conv_out = nn.Conv2d(img_ch // 4, output_ch, kernel_size=1, stride=1, padding=0)  # 1x28x28
        #
        # self.conv_up_blocks = nn.Sequential(up_conv1, conv1, up_conv2, conv2, conv_out)

    def forward(self, x: Tensor) -> Tensor:
        # decoding path
        return self.conv_blocks(x)
        # return self.conv_up_blocks(x)


class AeConv(nn.Module):
    def __init__(self, img_ch=1, output_ch=1, n_encoder_ch=16, img_size: int = 32, deep: int = 3):
        super(AeConv, self).__init__()
        assert deep == 3
        self.n_encoder_ch = n_encoder_ch
        self.n_features_ch = self.n_encoder_ch * 4
        self.n_features_size = int(np.ceil(img_size / (2 * (deep-1))))
        self.encoder = EncoderConv(in_ch=img_ch, encoder_ch=self.n_encoder_ch, deep=deep)
        self.decoder = DecoderConv(output_ch=output_ch, img_ch=self.n_features_ch, deep=deep)

    def forward(self, x: Tensor) -> Tensor:
        features = self.encoder(x)
        x_out = self.decoder(features)
        return x_out, features


class Discriminator(nn.Module):
    def __init__(self, input_ch=1, in_conv_ch: Optional[int] = 8, img_size: int = 28,
                 input_norm_type: str = None, fc_norm_type: str = None,
                 n_fc_layers: Optional[List[int]] = None, deep_conv_net: int = 2, reduce_validity: bool = False):
        super(Discriminator, self).__init__()
        self.n_fc_layers = [512, 256, 1] if n_fc_layers is None else n_fc_layers
        self.in_conv_ch = in_conv_ch
        if self.in_conv_ch is not None:
            self.conv_encoder = EncoderConv(in_ch=input_ch, encoder_ch=self.in_conv_ch, last_down=False,
                                            deep=deep_conv_net)
            # scale_factor = 2 ** (self.encoder.deep - 1)
            # scale_factor = 4
            scale_factor = 2 ** self.conv_encoder.deep
            enc_size = img_size // scale_factor
            out_conv_ch = self.conv_encoder.out_ch
            # out_conv_ch = self.in_conv_ch * scale_factor
        else:
            self.conv_encoder = nn.Identity()
            enc_size = img_size
            out_conv_ch = input_ch

        if reduce_validity:
            self.fc_out = nn.Sequential(nn.LeakyReLU(0.2, inplace=True), nn.Linear(self.n_fc_layers[-1], 1))
        else:
            self.fc_out = nn.Identity()

        self.input_norm = get_norm_layer(input_norm_type, input_ch, img_size)

        in_fc_ch = out_conv_ch * (enc_size ** 2)
        self.adv_fc = []
        for fc_out_ch in self.n_fc_layers[: -1]:
            layer_fc = nn.Sequential(nn.Linear(in_fc_ch, fc_out_ch),
                                     get_norm_layer(fc_norm_type, fc_out_ch),
                                     nn.LeakyReLU(0.2, inplace=True))
            self.adv_fc.append(layer_fc)
            in_fc_ch = fc_out_ch

        self.adv_fc.append(nn.Linear(self.n_fc_layers[-2], self.n_fc_layers[-1]))
        self.adv_fc = nn.Sequential(*self.adv_fc)


    def forward(self, x: Tensor) -> DiscriminatorBatch:
        x_norm = self.input_norm(x)
        conv_features = self.conv_encoder(x_norm)
        conv_features_flat = conv_features.view(conv_features.shape[0], -1)
        fc_features = self.adv_fc(conv_features_flat)
        validity = self.fc_out(fc_features)
        out_features = [conv_features, fc_features]
        return DiscriminatorBatch(validity=validity, features=out_features)


