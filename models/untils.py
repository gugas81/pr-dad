
from torch import Tensor
import torch.nn as nn
from typing import Optional

from models.ada_in_layer import AdaInBlock


def get_norm_layer(norm_type: str, n_channel: int,
                   img_size: Optional[int] = None,
                   dim_latent: Optional[int] = None) -> nn.Module:
    if img_size is not None:
        is_2d = True
        assert img_size

    if norm_type is None:
        norm_layer = nn.Identity()
    elif norm_type == 'batch_norm':
        norm_layer = nn.BatchNorm2d(n_channel) if is_2d else nn.BatchNorm1d(n_channel)
    elif norm_type == 'layer_norm':
        norm_layer = nn.LayerNorm((n_channel, img_size, img_size)) if is_2d else nn.LayerNorm(n_channel)
    elif norm_type == 'instance_norm':
        norm_layer = nn.InstanceNorm2d((n_channel, img_size, img_size)) if is_2d else nn.InstanceNorm1d(n_channel)
    elif norm_type == 'ada-in':
        norm_layer = AdaInBlock(n_channel, dim_latent)
    else:
        raise NameError(f'Non valid type: {norm_type}')
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