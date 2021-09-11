import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from models.seq_blocks import EncoderConv, DecoderConv


class AeConv(nn.Module):
    def __init__(self, img_ch: int = 1, output_ch: int = 1, n_encoder_ch: int = 16, img_size: int = 32, deep: int = 3,
                 down_pool: str = 'avrg_pool', active_type: str = 'leakly_relu', up_mode: str = 'bilinear',
                 features_sigmoid_active: bool = True):
        super(AeConv, self).__init__()
        self.features_sigmoid_active = features_sigmoid_active
        self.n_encoder_ch = n_encoder_ch
        scale_factor = 2 ** (deep-1)
        self.n_features_ch = self.n_encoder_ch * scale_factor
        self.n_features_size = int(np.ceil(img_size / scale_factor))
        self._encoder = EncoderConv(in_ch=img_ch, encoder_ch=self.n_encoder_ch, deep=deep,
                                    active_type=active_type, down_pool=down_pool)
        self._decoder = DecoderConv(output_ch=None, img_ch=self.n_features_ch, deep=deep,
                                    up_mode=up_mode, active_type=active_type)
        self.out_layer = nn.Conv2d(self._decoder.ch_out[-1], output_ch, kernel_size=1, stride=1, padding=0)

    def encode(self, x: Tensor) -> Tensor:
        features = self._encoder(x)
        if self.features_sigmoid_active:
            features = torch.sigmoid(features)
        return features

    def decode(self,  features: Tensor) -> Tensor:
        x_out = self._decoder(features)
        x_out = self.out_layer(x_out)
        # x_out = torch.sigmoid(x_out)
        # x_out = torch.clip(x_out, -1.0, 1.0)
        return x_out

    def forward(self, x: Tensor) -> (Tensor, Tensor):
        features = self.encode(x)
        x_out = self.decode(features)

        return x_out, features





