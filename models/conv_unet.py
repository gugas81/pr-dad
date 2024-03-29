import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional, List, Tuple

from models.seq_blocks import EncoderConv, DecoderConv
from models.layers import SpatialAttentionBlock
from models.untils import ConcatList


class UNetConv(nn.Module):
    def __init__(self,
                 img_size: int,
                 img_ch: int = 1,
                 output_ch: int = 1, 
                 n_encoder_ch: int = 16,
                 deep: int = 3,
                 in_ch_features: Optional[int] = None,
                 up_mode: str = 'nearest',
                 skip_input: bool = False,
                 active_type: str = 'leakly_relu',
                 down_pool: str = 'avrg_pool',
                 features_sigmoid_active: bool = False,
                 special_attention: bool = False):
        super(UNetConv, self).__init__()
        self.features_sigmoid_active = features_sigmoid_active
        self.skip_input = skip_input
        self._in_ch_features = in_ch_features
        self.special_attention = special_attention
        self._encoder = EncoderConv(in_ch=img_ch, encoder_ch=n_encoder_ch, deep=deep,
                                    active_type=active_type, down_pool=down_pool)
        self.att_blocks = ConcatList()
        if self.special_attention:
            self.att_blocks.extend([SpatialAttentionBlock(enc_ch_) for enc_ch_ in self._encoder.out_ch])

        self._n_features_ch = self._encoder.out_ch[-1]
        skip_connect_decoder = self._encoder.out_ch[::-1]
        if self._in_ch_features:
            skip_connect_decoder[0] = self._in_ch_features
            if self.special_attention:
                self._att_features_block = SpatialAttentionBlock(self._in_ch_features)
            else:
                self._att_features_block = None
        else:
            skip_connect_decoder[0] = 0
        self._decoder = DecoderConv(output_ch=None,
                                    img_ch=self._n_features_ch,
                                    deep=deep,
                                    skip_connect_ch=skip_connect_decoder,
                                    up_mode=up_mode,
                                    active_type=active_type)

        self.conv_out = nn.Conv2d(self._decoder.ch_outs[-1], output_ch, kernel_size=1, stride=1, padding=0)

        scale_factor = 2 ** (deep - 1)
        self._n_features_size = int(np.ceil(img_size / scale_factor))

    def decode(self, enc_features: List[torch.Tensor]) -> Tensor:
        out_features = None
        for ind, (enc_features_layer,  decoder_layer) in enumerate(zip(enc_features, self._decoder.get_layers())):
            in_features = enc_features_layer
            if out_features is not None:
                in_features = torch.cat((in_features, out_features), dim=1)
            out_features = decoder_layer(in_features)
        return out_features

    def forward(self, x: Tensor, in_features: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # encoding path
        encoder_features = self._encoder(x, use_residual=True)

        decoder_features = self.get_decoder_features(encoder_features, in_features)

        out_features = self.decode(decoder_features)
        x_out = self.conv_out(out_features)

        if self.skip_input:
            x_out = x_out + x

        return x_out, encoder_features, decoder_features, None

    def get_decoder_features(self, encoder_features, in_features):
        if self.features_sigmoid_active:
            encoder_features[-1] = torch.sigmoid(encoder_features[-1])
        if self.special_attention:
            encoder_features = [enc_f * att_block(enc_f) for enc_f, att_block in
                                zip(encoder_features[1:], self.att_blocks)]
            in_features = in_features * self._att_features_block(in_features)
        else:
            encoder_features = encoder_features[1:]
        if in_features is not None and self._in_ch_features:
            encoder_features[-1] = torch.cat((encoder_features[-1], in_features), dim=1)
        encoder_features = encoder_features[::-1]
        return encoder_features

    @property
    def n_enc_features_ch(self) -> int:
        return self._n_features_ch

    @property
    def n_features_size(self) -> int:
        return self._n_features_size

