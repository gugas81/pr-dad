import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional, List
from models.seq_blocks import EncoderConv, DecoderConv


class UNetConv(nn.Module):
    def __init__(self, img_ch: int = 1, output_ch: int = 1, n_encoder_ch=16, deep: int = 3,
                 in_ch_features: Optional[int] = None, up_mode: str = 'nearest', skip_input: bool = False,
                 active_type: str = 'leakly_relu', down_pool: str = 'avrg_pool', features_sigmoid_active: bool = False):
        super(UNetConv, self).__init__()
        self.features_sigmoid_active = features_sigmoid_active
        self.skip_input = skip_input
        self._in_ch_features = in_ch_features
        self._encoder = EncoderConv(in_ch=img_ch, encoder_ch=n_encoder_ch, deep=deep,
                                    active_type=active_type, down_pool=down_pool)
        self.n_features_ch = self._encoder.out_ch[-1]
        skip_connect_decoder = self._encoder.out_ch[::-1]
        if self._in_ch_features:
            skip_connect_decoder[0] = self._in_ch_features
        else:
            skip_connect_decoder[0] = 0
        self._decoder = DecoderConv(output_ch=None, img_ch=self.n_features_ch, deep=deep,
                                    skip_connect_ch=skip_connect_decoder, up_mode=up_mode, active_type=active_type)

        self.conv_out = nn.Conv2d(self._decoder.ch_out[-1], output_ch, kernel_size=1, stride=1, padding=0)

    def decode(self, encoder_features: List[torch.Tensor]) -> Tensor:
        out_features = None
        for enc_encoder_features_layer,  decoder_layer in zip(encoder_features, self._decoder.get_layers()):
            in_features = enc_encoder_features_layer
            if out_features is not None:
                in_features = torch.cat((in_features, out_features), dim=1)
            out_features = decoder_layer(in_features)
        return out_features

    def forward(self, x: Tensor, features: Optional[Tensor] = None) -> Tensor:
        # encoding path
        encoder_features = self._encoder(x, use_residual=True)

        if self.features_sigmoid_active:
            encoder_features[-1] = torch.sigmoid(encoder_features[-1])

        if features is not None and self._in_ch_features:
            encoder_features[-1] = torch.cat((encoder_features[-1], features), dim=1)
        encoder_features = encoder_features[1:][::-1]
        x_out = self.decode(encoder_features)
        x_out = self.conv_out(x_out)
        if self.skip_input:
            x_out = x_out + x
        return x_out

