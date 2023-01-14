from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from models.seq_blocks import EncoderConv, DecoderConv, MlpNet
from models.layers import ConvBlock
from models.base_autoencoder import BaseAe
from models.untils import BlockList


class AttDictionary(nn.Module):
    def __init__(self, n_ch: int, img_size: int):
        super().__init__()
        self.dictionary = nn.Parameter(torch.randn((n_ch, img_size, img_size)), requires_grad=True)

    def forward(self, features: Tensor) -> Tensor:
        return features * self.dictionary


class MulDictionary(nn.Module):
    def __init__(self, dim: int, img_size: int):
        super().__init__()
        self.dim = dim
        self.img_size = img_size
        self.dictionary = torch.empty(self.dim, self.img_size, self.img_size)
        nn.init.xavier_uniform_(self.dictionary, gain=0.01)
        self.dictionary = nn.Parameter(self.dictionary, requires_grad=True)

    def forward(self, coeff: Tensor) -> Tensor:
        return torch.matmul(coeff, self.dictionary.view(1, self.dim, -1)).view(-1, coeff.shape[1], self.img_size,
                                                                               self.img_size)


class MapToCoeff(nn.Module):
    def __init__(self, in_ch: int, img_size: int, out_coeff: int, out_ch: int, deep_mlp: int = 3):
        super().__init__()
        self.out_coeff = out_coeff
        self.out_ch = out_ch
        self.in_features = in_ch * (img_size ** 2)

        self.mlp_maps = MlpNet(in_ch=self.in_features,
                               deep=deep_mlp,
                               out_ch=self.out_coeff * self.out_ch,
                               multy_coeff=0.5)

    def forward(self, features: Tensor) -> Tensor:
        coeff = self.mlp_maps(features.view(-1, self.in_features))
        return coeff.view(-1, self.out_ch, self.out_coeff)


class AeConv(BaseAe):
    def __init__(self, img_ch: int = 1, output_ch: int = 1, n_encoder_ch: int = 16, img_size: int = 32, deep: int = 3,
                 n_enc_features: int = None, n_dec_features: int = None,
                 down_pool: str = 'avrg_pool', active_type: str = 'leakly_relu', up_mode: str = 'bilinear',
                 features_sigmoid_active: bool = True, use_dictionary: bool = False, dict_len: int = 16,
                 multi_scale_out: bool = False):
        super(AeConv, self).__init__(img_size=img_size, in_ch=img_ch, deep=deep)
        self.use_dictinary = use_dictionary
        self.multi_scale_out = multi_scale_out
        self.features_sigmoid_active = features_sigmoid_active
        self.n_encoder_ch = n_encoder_ch
        scale_factor = 2 ** (deep - 1)
        self._n_enc_features_ch = self.n_encoder_ch * scale_factor if n_enc_features is None else n_enc_features
        self._n_dec_features_ch = self.n_enc_features_ch if n_dec_features is None else n_dec_features
        self._n_features_size = int(np.ceil(img_size / scale_factor))
        # padding_mode = 'replicate'
        if self.use_dictinary:
            self.dictionary: Optional[MulDictionary] = MulDictionary(dict_len, self.n_features_size)
            self.map_to_coeff = MapToCoeff(in_ch=self.n_enc_features_ch,
                                           img_size=self.n_features_size,
                                           out_coeff=dict_len,
                                           out_ch=self.n_dec_features_ch)
        else:
            assert self.n_enc_features_ch == self.n_dec_features_ch
            self.dictionary: Optional[MulDictionary] = None
        self._encoder = EncoderConv(in_ch=img_ch, encoder_ch=self.n_encoder_ch, deep=deep,
                                    active_type=active_type, down_pool=down_pool, padding_mode='zeros')
        self._decoder = DecoderConv(output_ch=None, img_ch=self.n_dec_features_ch, deep=deep,
                                    up_mode=up_mode, active_type=active_type,
                                    multi_scale_out=multi_scale_out)
        if multi_scale_out:
            self.out_layer = BlockList()
            for out_ch in self._decoder.ch_outs:
                self.out_layer.append(ConvBlock(out_ch, output_ch, active_type=active_type))

        else:
            self.out_layer = nn.Conv2d(self._decoder.ch_outs[-1], output_ch, kernel_size=1, stride=1, padding=0)

    @property
    def n_enc_features_ch(self) -> int:
        return self._n_enc_features_ch

    @property
    def n_dec_features_ch(self) -> int:
        return self._n_dec_features_ch

    @property
    def n_features_size(self) -> int:
        return self._n_features_size

    def encode(self, x: Tensor) -> Tensor:
        features = self._encoder(x)

        if self.features_sigmoid_active:
            features = torch.sigmoid(features)

        return features

    def decode(self, features: Tensor) -> Tensor:
        x_out_decoder = self._decoder(features, use_residual=self.multi_scale_out)
        if self.multi_scale_out:
            x_out = []
            for out_layer, x_tensor in zip(self.out_layer, x_out_decoder):
                x_out.append(out_layer(x_tensor))
        else:
            x_out = self.out_layer(x_out_decoder)

        return x_out

    def get_dictionary(self) -> Optional[Tensor]:
        return self.dictionary.dictionary if self.use_dictinary else None

    def apply_dictionary(self, coeff: Tensor) -> Tensor:
        return self.dictionary(coeff)

    def map_to_dec_features(self, enc_features: Tensor) -> (Tensor, Tensor):
        if self.use_dictinary:
            coeff = self.map_to_coeff(enc_features)
            dec_features = self.apply_dictionary(coeff)
        else:
            dec_features = enc_features
            coeff = None

        return dec_features, coeff

    def forward(self, x: Tensor) -> (Tensor, Tensor, Tensor, Tensor):
        enc_features = self.encode(x)
        dec_features, coeff = self.map_to_dec_features(enc_features)
        x_out = self.decode(dec_features)

        return x_out, enc_features, dec_features, coeff
