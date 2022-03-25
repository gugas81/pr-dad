import math
import torch
from torch import Tensor
import torch.nn as nn
from typing import List, Optional, Union, Sequence

from models.untils import ConcatList
from models.seq_blocks import MlpNet


class MlpMultyHeadsPredictor(nn.Module):
    def __init__(self, in_ch: int,
                 n_features: int,
                 features_dim_2d: Sequence[int],
                 deep_emb: int = 3,
                 deep_predict: int = 3,
                 emd_dim: Optional[int] = None,
                 active_type: str = 'prelu',
                 active_ch: bool = True,
                 norm_type: Optional[str] = None,
                 emb_multy_coeff: float = 2.0,
                 predict_multy_coeff: float = 0.5,
                 use_dropout: bool = False,
                 ):
        super(MlpMultyHeadsPredictor, self).__init__()

        self.emb_net = MlpNet(in_ch=in_ch,
                              deep=deep_emb,
                              multy_coeff=emb_multy_coeff,
                              out_ch=emd_dim,
                              use_dropout=use_dropout,
                              norm_type=norm_type,
                              active_ch=active_ch)

        self.f_predict_heads = ConcatList()
        self._features_dim_2d = features_dim_2d
        self._n_features = n_features
        features_dim = self._features_dim_2d[0] * self._features_dim_2d[1]
        for ind_net in range(self._n_features):
            predict_net = MlpNet(in_ch=emd_dim,
                                 deep=deep_predict,
                                 multy_coeff=predict_multy_coeff,
                                 out_ch=features_dim,
                                 use_dropout=use_dropout,
                                 norm_type=norm_type,
                                 active_type=active_type,
                                 active_ch=active_ch)
            self.f_predict_heads.append(predict_net)

    def forward(self, x: Tensor) -> Union[Tensor, List[Tensor]]:
        emb = self.emb_net(x)
        features_predict = self.f_predict_heads(emb, concat_out='stack')
        features_predict = features_predict.view(-1, self._n_features, *self._features_dim_2d)
        return features_predict

