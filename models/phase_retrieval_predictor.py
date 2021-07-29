import torch
from torch import Tensor
import torch.nn as nn

from models.layers import FcBlock, ConvBlock, ResBlock
from models.conv_unet import UNetConv


class PhaseRetrievalPredictor(nn.Module):
    # class TypeRecon(Enum):

    def __init__(self, use_dropout: bool = False, im_img_size: int = 28, inter_ch: int = 1, out_ch: int = 1,
                 out_img_size: int = 32,
                 fc_multy_coeff: int = 1, use_bn: bool = False, fft_norm: str = "ortho", deep_fc: int = 4,
                 deep_conv: int = 2,
                 predict_type: str = 'spectral', conv_type: str = 'ConvBlock',
                 active_type: str = 'leakly_relu', features_sigmoid_active: bool = False):
        super(PhaseRetrievalPredictor, self).__init__()
        self.features_sigmoid_active = features_sigmoid_active
        self.im_img_size = im_img_size
        self.out_img_size = out_img_size
        self._predict_type = predict_type

        self.out_ch = out_ch
        self.int_ch = inter_ch
        self.fc_multy_coeff = fc_multy_coeff
        self.in_features = self.im_img_size ** 2
        self.inter_features = self.int_ch * self.out_img_size ** 2

        self.out_features = self.out_ch * self.out_img_size ** 2
        self._fft_norm = fft_norm

        if self._predict_type == 'spectral':
            out_fc_features = 2 * self.inter_features
        elif self._predict_type == 'phase':
            out_fc_features = self.inter_features
        else:
            raise NameError(f'Not supported type_recon: {predict_type}')

        out_fc = self.in_features
        in_fc = self.in_features
        self.fc_blocks = []
        for ind in range(deep_fc):
            if ind == deep_fc - 1:
                out_fc = out_fc_features
            fc_block = FcBlock(in_fc, out_fc, use_dropout=use_dropout, use_bn=use_bn)
            in_fc = out_fc

            out_fc *= self.fc_multy_coeff
            self.fc_blocks.append(fc_block)

        self.fc_blocks = nn.Sequential(*self.fc_blocks)

        self._build_conv_blocks(conv_type, deep_conv, active_type=active_type)

    def _build_conv_blocks(self, conv_type: str, deep_conv: int, active_type: str = 'leakly_relu'):
        if conv_type == 'ConvBlock':
            conv_block_class = ConvBlock
        elif conv_type == 'ResBlock':
            conv_block_class = ResBlock
        elif conv_type == 'Unet':
            conv_block_class = UNetConv
        else:
            raise NameError(f'Non valid conv_type: {conv_type}')

        if conv_type == 'Unet':
            self.conv_blocks = UNetConv(img_ch=self.int_ch, output_ch=self.out_ch,
                                        up_mode='bilinear', active_type=active_type, down_pool='avrg_pool')
        else:
            in_conv = self.int_ch
            out_conv = 2 * in_conv
            self.conv_blocks = []
            for ind in range(deep_conv):
                conv_block = conv_block_class(in_conv, out_conv, active_type=active_type)
                in_conv = out_conv
                self.conv_blocks.append(conv_block)
            conv_out = nn.Conv2d(out_conv, self.out_ch, kernel_size=1, stride=1, padding=0)
            self.conv_blocks.append(conv_out)
            self.conv_blocks = nn.Sequential(*self.conv_blocks)

    def forward(self, magnitude: Tensor) -> (Tensor, Tensor):
        if self._predict_type == 'spectral':
            out_features, intermediate_features = self._spectral_pred(magnitude)
        elif self._predict_type == 'phase':
            out_features = self._angle_pred(magnitude)
            intermediate_features = out_features
            out_features = out_features.real

        if self.features_sigmoid_active:
            out_features = torch.sigmoid(out_features)

        return out_features, intermediate_features

    def _angle_pred(self,  magnitude: Tensor) -> Tensor:
        magnitude = torch.fft.fftshift(magnitude, dim=(-2, -1))
        phase = self.conv_blocks(magnitude)
        exp_phase = torch.exp(torch.view_as_complex(torch.stack([torch.zeros_like(magnitude), phase], -1)))
        spectral = magnitude * exp_phase
        spectral = torch.fft.ifftshift(spectral, dim=(-2, -1))
        out_features = torch.fft.ifft2(spectral, norm=self._fft_norm)
        return out_features

    def _spectral_pred(self, magnitude: Tensor) -> (Tensor, Tensor):
        x_float = magnitude.view(-1, self.in_features)
        fc_features = self.fc_blocks(x_float)

        spectral = fc_features.view(-1, self.int_ch, self.out_img_size, self.out_img_size, 2)
        spectral = torch.view_as_complex(spectral)
        intermediate_features = torch.fft.ifft2(spectral, norm=self._fft_norm)

        out_features = self.conv_blocks(intermediate_features.real)

        return out_features, intermediate_features




