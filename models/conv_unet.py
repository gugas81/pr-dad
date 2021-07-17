import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional, List
from models.seq_blocks import EncoderConv, DecoderConv


class UNetConv(nn.Module):
    def __init__(self, img_ch: int = 1, output_ch: int = 1, n_encoder_ch=16, deep: int = 3, in_ch_features: Optional[int] = None):
        super(UNetConv, self).__init__()

        self.encoder = EncoderConv(in_ch=img_ch, encoder_ch=n_encoder_ch, deep=deep)
        self.n_features_ch = self.encoder.out_ch[-1]
        # if in_ch_features:
        #     self.n_features_ch += in_ch_features
        skip_connect_decoder = self.encoder.out_ch[::-1]
        if in_ch_features:
            skip_connect_decoder[0] = in_ch_features
        else:
            skip_connect_decoder[0] = 0
        self.decoder = DecoderConv(output_ch=None, img_ch=self.n_features_ch, deep=deep,
                                   skip_connect_ch=skip_connect_decoder)

        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        #
        # self.conv_1 = ConvBlock(ch_in=img_ch, ch_out=16)
        # self.conv_2 = ConvBlock(ch_in=16, ch_out=32)
        # self.conv_3 = ConvBlock(ch_in=32, ch_out=64)
        #
        # inter_ch = 64 + in_ch_features
        # self.up_3 = UpConvBlock(ch_in=64, ch_out=32)
        # self.up_conv_3 = ConvBlock(ch_in=64, ch_out=32)
        #
        # self.up_2 = UpConvBlock(ch_in=32, ch_out=16)
        # self.up_conv_2 = ConvBlock(ch_in=32, ch_out=16)

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
        return x_out
        # x1 = self.conv_1(x) # 16x27x27
        #
        # x2 = self.maxpool(x1)# 16x14x14
        # x2 = self.conv_2(x2) # 32x14x14
        #
        # x3 = self.maxpool(x2)# 32x7x7
        # inter_features = self.conv_3(x3)# 64x7x7
        #
        #
        #
        # d3 = self.up_3(x3) # 32x14x14
        # d3 = torch.cat((x2, d3), dim=1)  # 32x14x14
        # d3 = self.up_conv_3(d3)  # 32x14x14
        #
        # d2 = self.up_2(d3) # 16x28x28
        # d2 = torch.cat((x1, d2), dim=1)  # 16x28x28
        # d2 = self.up_conv_2(d2) # 16x28x28
        #
        # d1 = self.conv_out(d2) # 1x28x28
        # return d1
