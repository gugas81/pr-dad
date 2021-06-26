
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

from models.layers import FcBlock, ConvBlock, ResBlock


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
            if ind == deep_fc - 1:
                out_fc = out_fc_features
            fc_block = FcBlock(in_fc, out_fc, use_dropout=use_dropout, use_bn=use_bn)
            in_fc = out_fc

            out_fc *= self.multy_coeff
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

