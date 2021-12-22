import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional, List, Union


class BlockList(nn.ModuleList):
    def __call__(self, x: Tensor, use_residual: bool = False):
        if use_residual:
            return self._get_resid(x)
        else:
            return self._get_sequential(x)

    def _get_resid(self, x: Tensor) -> List[Tensor]:
        results = [x]
        for block in self:
            x = block(x)
            results.append(x)
        return results

    def _get_sequential(self, x: Tensor) -> Tensor:
        for block in self:
            x = block(x)
        return x


class ConcatList(nn.ModuleList):
    def forward(self, x: Tensor) -> List[Tensor]:
        out = [block(x) for block in self]
        # out = torch.stack(out, 1)
        return out


def get_norm_layer(name_type: str, input_ch: int, img_size: Optional[Union[int, List[int]]] = None,
                   is_2d: bool = True, affine: bool = True) -> nn.Module:
    name_type = name_type.lower() if name_type else None
    if isinstance(img_size, int) and is_2d:
        img_size = (img_size, img_size)

    if name_type is None:
        norm_layer = nn.Identity()
    elif name_type == 'batch_norm':
        norm_layer = nn.BatchNorm2d(input_ch, affine=affine) if is_2d else nn.BatchNorm1d(input_ch, affine=affine)
    elif name_type == 'layer_norm':
        norm_layer = nn.LayerNorm((input_ch, img_size[0], img_size[1]), elementwise_affine=affine) if is_2d else \
            nn.LayerNorm(input_ch, elementwise_affine=affine)
    elif name_type == 'instance_norm':
        norm_layer = nn.InstanceNorm2d(input_ch, affine=affine) if is_2d else nn.InstanceNorm1d(input_ch, affine=affine)
    else:
        raise NameError(f'Non valid type: {name_type}')
    return norm_layer


def get_activation(name: str, num_parameters: int = 1) -> nn.Module:
    name = name.lower()
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'leakly_relu':
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    elif name == 'prelu':
        return nn.PReLU(num_parameters=num_parameters)
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