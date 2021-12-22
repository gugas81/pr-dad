import math
import torch
from torch import Tensor
import torch.nn as nn
from typing import List, Optional

from models.untils import ConcatList
from models.seq_blocks import MlpNet


class MlpFeaturesEmedings(nn.Module):
    def __init__(self, in_ch: int,
                 n_features: int,
                 ch_features: int,
                 deep_emb: int = 3,
                 deep_predict: int = 3,
                 emd_dim: Optional[int] = None,
                 active_type: str = 'prelu',
                 active_ch: bool = True,
                 norm_type: Optional[str] = None,
                 emb_multy_coeff: float = 2.0,
                 predict_multy_coeff: float = 2.0,
                 use_dropout: bool = False,
                 ):

        self.emb_net = MlpNet(in_ch=in_ch,
                              deep=deep_emb,
                              multy_coeff=emb_multy_coeff,
                              out_ch=emd_dim,
                              use_dropout=use_dropout,
                              norm_type=norm_type,
                              active_ch=active_ch)

        self.f_predict_nets = ConcatList()
        for ind_net in range(n_features):
            predict_net = MlpNet(in_ch=emd_dim,
                                 deep=deep_predict,
                                 multy_coeff=predict_multy_coeff,
                                 out_ch=ch_features,
                                 use_dropout=use_dropout,
                                 norm_type=norm_type,
                                 active_type=active_type,
                                 active_ch=active_ch)
            self.f_predict_nets.append(predict_net)

    def forward(self, x: Tensor) -> List[Tensor]:
        emb = self.emb_net(x)
        features_predict = self.f_predict_nets(emb)
        return features_predict

