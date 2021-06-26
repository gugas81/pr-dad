
from torch import Tensor
import torch.nn as nn
from typing import Optional


def get_norm_layer(input_norm_type: str, input_ch: int, img_size: Optional[int] = None) -> nn.Module:
    if img_size is not None:
        is_2d = True
        assert img_size

    if input_norm_type is None:
        norm_layer = nn.Identity()
    elif input_norm_type == 'batch_norm':
        norm_layer = nn.BatchNorm2d(input_ch) if is_2d else nn.BatchNorm1d(input_ch)
    elif input_norm_type == 'layer_norm':
        norm_layer = nn.LayerNorm((input_ch, img_size, img_size)) if is_2d else nn.LayerNorm(input_ch)
    elif input_norm_type == 'instance_norm':
        norm_layer = nn.InstanceNorm2d((input_ch, img_size, img_size)) if is_2d else nn.InstanceNorm1d(input_ch)
    else:
        raise NameError(f'Non valid type: {input_norm_type}')
    return norm_layer