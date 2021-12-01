import math
import torch
from torch import Tensor
import torchvision
import torch.nn as nn
from typing import List
import torchjpeg.dct as jpeg_dct

from common import ConfigTrainer
import common.utils as utils

from models.untils import BlockList, get_norm_layer
from models.layers import FcBlock, ConvBlock, ResBlock
from models.conv_unet import UNetConv


class PhaseRetrievalPredictor(nn.Module):
    def __init__(self,
                 config: ConfigTrainer,
                 out_ch: int,
                 out_img_size: int):
        super(PhaseRetrievalPredictor, self).__init__()
        self._config: ConfigTrainer = config
        self.out_img_size = out_img_size
        self.out_ch = out_ch

        if self._config.predict_out == 'features':
            if self._config.n_inter_features is None:
                self.inter_ch = out_ch
                if self._config.predict_type == 'spectral':
                    self.inter_ch *= 2
            else:
                self.inter_ch = self._config.n_inter_features
        else:
            self.inter_ch = self._config.predict_img_int_features_multi_coeff * out_ch

        self.inter_mag_out_size = utils.get_magnitude_size_2d(out_img_size, self._config.add_pad,
                                                              use_rfft=(self._config.use_rfft and not self._config.use_dct))

        input_mag_size_2d = utils.get_magnitude_size_2d(self._config.image_size, self._config.add_pad,
                                                        use_rfft=(self._config.use_rfft and not self._config.use_dct_input))
        self.in_features = input_mag_size_2d[0]*input_mag_size_2d[1]

        self.inter_features = self.inter_ch * self.inter_mag_out_size[0] * self.inter_mag_out_size[1]

        if self._config.deep_predict_fc is None:
            deep_fc = int(math.floor(math.log(self.inter_features / self.in_features,
                                              self._config.predict_fc_multy_coeff))) + 1
            deep_fc = max(3, deep_fc)
        else:
            deep_fc = self._config.deep_predict_fc

        self.out_features = self.out_ch * self.inter_mag_out_size[0] * self.inter_mag_out_size[1]

        if self._config.predict_type == 'spectral':
            if self._config.use_dct:
                out_fc_features = self.inter_features
            else:
                out_fc_features = 2 * self.inter_features
        elif self._config.predict_type == 'phase':
            out_fc_features = self.inter_features
        else:
            raise NameError(f'Not supported type_recon: {self._config.predict_type}')

        in_fc = self.in_features
        self.input_norm = get_norm_layer(name_type=self._config.magnitude_norm,
                                         input_ch=self.in_features,
                                         img_size=input_mag_size_2d,
                                         is_2d=True,
                                         affine=True)
        self.fc_blocks = BlockList()
        for ind in range(deep_fc):
            if ind == deep_fc - 1:
                out_fc = out_fc_features
            else:
                out_fc = in_fc * self._config.predict_fc_multy_coeff

            fc_block = FcBlock(in_fc, out_fc,
                               use_dropout=self._config.use_dropout_enc_fc,
                               norm_type=self._config.norm_fc,
                               active_type=self._config.activation_fc_enc,
                               active_params=out_fc if self._config.activation_fc_ch_params else 1)
            in_fc = out_fc

            self.fc_blocks.append(fc_block)

        self.weights_fc: List[nn.Parameter] = [fc_block.fc_seq[0].weight for fc_block in self.fc_blocks]
        self._build_conv_blocks(self._config.predict_conv_type,
                                self._config.deep_predict_conv,
                                active_type=self._config.activation_enc)

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
            self.conv_blocks = UNetConv(img_ch=self.inter_ch, output_ch=self.out_ch,
                                        up_mode='bilinear', active_type=active_type, down_pool='avrg_pool')
        else:
            in_conv = self.inter_ch

            self.conv_blocks = BlockList()
            for ind in range(deep_conv):
                out_conv = self._config.predict_conv_multy_coeff * in_conv
                conv_block = conv_block_class(in_conv, out_conv, active_type=active_type)
                in_conv = out_conv
                self.conv_blocks.append(conv_block)
            conv_out = nn.Conv2d(out_conv, self.out_ch, kernel_size=1, stride=1, padding=0)
            self.conv_blocks.append(conv_out)

    def forward(self, magnitude: Tensor) -> (Tensor, Tensor):
        magnitude = self.input_norm(magnitude)
        if self._config.predict_type == 'spectral':
            out_features, intermediate_features = self._spectral_pred(magnitude)
        elif self._config.predict_type == 'phase':
            out_features = self._angle_pred(magnitude)
            intermediate_features = out_features
            out_features = out_features.real

        if self._config.features_sigmoid_active:
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
        mag_flatten = magnitude.view(-1, self.in_features)
        fc_features = self.fc_blocks(mag_flatten)
        if self._config.use_dct:
            spectral = fc_features.view(-1, self.inter_ch, self.inter_mag_out_size[0], self.inter_mag_out_size[1])
            intermediate_features = 4.0 * jpeg_dct.block_idct(spectral)
            out_features = intermediate_features
        else:
            spectral = fc_features.view(-1, self.inter_ch, self.inter_mag_out_size[0], self.inter_mag_out_size[1], 2)
            spectral = torch.view_as_complex(spectral)

            if self._config.use_rfft:
                intermediate_features = torch.fft.irfft2(spectral, (self.inter_mag_out_size[0], self.inter_mag_out_size[0]),
                                                         norm=self._config.fft_norm)
                out_features = intermediate_features

            else:
                intermediate_features = torch.fft.ifft2(spectral, norm=self._config.fft_norm)
                out_features = intermediate_features.real

        out_features = torchvision.transforms.functional.center_crop(out_features, self.out_img_size)
        out_features = self.conv_blocks(out_features)

        return out_features, intermediate_features




