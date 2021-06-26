import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from models.seq_blocks import EncoderConv, DecoderConv


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





