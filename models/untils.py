
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


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'leakly_relu':
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    else:
        raise NameError(f'Non valid activation type: {name}')


def get_pool_2x2(name: str) -> nn.Module:
    name = name.lower()
    if name == 'avrg_pool':
        return nn.AvgPool2d(kernel_size=2, stride=2)
    elif name == 'max_pool':
        return nn.MaxPool2d(kernel_size=2, stride=2)
    else:
        raise NameError(f'Non valid activation type: {name}')