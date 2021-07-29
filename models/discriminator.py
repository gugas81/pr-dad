from torch import Tensor
import torch.nn as nn
from typing import List, Optional
from models.untils import get_norm_layer
from common.data_classes import DiscriminatorBatch
from models.seq_blocks import EncoderConv


class Discriminator(nn.Module):
    def __init__(self, input_ch=1,
                 in_conv_ch: Optional[int] = 8,
                 img_size: int = 28,
                 input_norm_type: str = None,
                 fc_norm_type: str = None,
                 n_fc_layers: Optional[List[int]] = None,
                 deep_conv_net: int = 2,
                 reduce_validity: bool = False,
                 use_res_blocks: bool = False,
                 active_type: str = 'leakly_relu'):
        super(Discriminator, self).__init__()
        self.n_fc_layers = [512, 256, 1] if n_fc_layers is None else n_fc_layers
        self.in_conv_ch = in_conv_ch
        if self.in_conv_ch is not None:
            last_down = True
            self.conv_encoder = EncoderConv(in_ch=input_ch, encoder_ch=self.in_conv_ch, last_down=last_down,
                                            deep=deep_conv_net, use_res_blocks=use_res_blocks, down_pool='max_pool',
                                            active_type=active_type)
            scale_factor = (2 ** (self.conv_encoder.deep-1)) if last_down else (2 ** (self.conv_encoder.deep-2))
            enc_size = img_size // scale_factor
            out_conv_ch = self.conv_encoder.out_ch[-1]
        else:
            self.conv_encoder = nn.Identity()
            enc_size = img_size
            out_conv_ch = input_ch

        if reduce_validity:
            self.fc_out = nn.Sequential(nn.LeakyReLU(0.2, inplace=True), nn.Linear(self.n_fc_layers[-1], 1))
        else:
            self.fc_out = nn.Identity()

        self.input_norm = get_norm_layer(input_norm_type, input_ch, img_size)

        in_fc_ch = out_conv_ch * (enc_size ** 2)
        self.adv_fc = []
        for fc_out_ch in self.n_fc_layers[: -1]:
            layer_fc = nn.Sequential(nn.Linear(in_fc_ch, fc_out_ch),
                                     get_norm_layer(fc_norm_type, fc_out_ch),
                                     nn.LeakyReLU(0.2, inplace=True))
            self.adv_fc.append(layer_fc)
            in_fc_ch = fc_out_ch

        self.adv_fc.append(nn.Linear(self.n_fc_layers[-2], self.n_fc_layers[-1]))
        self.adv_fc = nn.Sequential(*self.adv_fc)

    def forward(self, x: Tensor) -> DiscriminatorBatch:
        x_norm = self.input_norm(x)
        conv_features = self.conv_encoder(x_norm)
        conv_features_flat = conv_features.view(conv_features.shape[0], -1)
        fc_features = self.adv_fc(conv_features_flat)
        validity = self.fc_out(fc_features)
        out_features = [conv_features, fc_features]
        return DiscriminatorBatch(validity=validity, features=out_features)

