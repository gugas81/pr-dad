import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional, List
from models.seq_blocks import EncoderConv, DecoderConv


class UNetConv(nn.Module):
    def __init__(self, img_ch: int = 1, output_ch: int = 1, n_encoder_ch=16, deep: int = 3,
                 in_ch_features: Optional[int] = None, mode: str = 'nearest', skip_input: bool = False):
        super(UNetConv, self).__init__()
        self.skip_input = skip_input
        self.encoder = EncoderConv(in_ch=img_ch, encoder_ch=n_encoder_ch, deep=deep)
        self.n_features_ch = self.encoder.out_ch[-1]
        skip_connect_decoder = self.encoder.out_ch[::-1]
        if in_ch_features:
            skip_connect_decoder[0] = in_ch_features
        else:
            skip_connect_decoder[0] = 0
        self.decoder = DecoderConv(output_ch=None, img_ch=self.n_features_ch, deep=deep,
                                   skip_connect_ch=skip_connect_decoder, mode=mode)

        self.conv_out = nn.Conv2d(self.decoder.ch_out[-1], output_ch, kernel_size=1, stride=1, padding=0)

    def decode(self, encoder_features: List[torch.Tensor]) -> Tensor:
        out_features = None
        for enc_encoder_features_layer,  decoder_layer in zip(encoder_features, self.decoder.get_layers()):
            in_features = enc_encoder_features_layer
            if out_features is not None:
                in_features = torch.cat((in_features, out_features), dim=1)
            out_features = decoder_layer(in_features)
        return out_features

    def forward(self, x: Tensor, features: Optional[Tensor] = None) -> Tensor:
        # encoding path
        encoder_features = self.encoder(x, use_residual=True)
        if features is not None:
            encoder_features[-1] = torch.cat((encoder_features[-1], features), dim=1)
        encoder_features = encoder_features[1:][::-1]
        x_out = self.decode(encoder_features)
        x_out = self.conv_out(x_out)
        if self.skip_input:
            x_out = x_out + x
        return x_out

