import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


# Constraints
# Input: [batch_size, in_channels, height, width]

# Scaled weight - He initialization
# "explicitly scale the weights at runtime"
class ScaleW:
    '''
    Constructor: name - name of attribute to be scaled
    '''

    def __init__(self, name: str):
        self.name = name

    def scale(self, module: nn.Module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * math.sqrt(2 / fan_in)

    @staticmethod
    def apply(module: nn.Module, name: str):
        '''
        Apply runtime scaling to specific module
        '''
        hook = ScaleW(name)
        weight = getattr(module, name)
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        del module._parameters[name]
        module.register_forward_pre_hook(hook)

    def __call__(self, module, whatever):
        weight = self.scale(module)
        setattr(module, self.name, weight)


# Quick apply for scaled weight
def quick_scale(module: nn.Module, name: str = 'weight'):
    ScaleW.apply(module, name)
    return module


# Uniformly set the hyperparameters of Linears
# "We initialize all weights of the convolutional, fully-connected, and affine transform layers using N(0, 1)"
# 5/13: Apply scaled weights
class SLinear(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()

        linear = nn.Linear(dim_in, dim_out)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = quick_scale(linear)

    def forward(self, x):
        return self.linear(x)


# "learned affine transform" A
class TransformWLatentToStyle(nn.Module):
    '''
    Learned affine transform A, this module is used to transform
    midiate vector w into a style vector
    '''
    def __init__(self, dim_latent: int, n_channel: int):
        super().__init__()
        self.mapping = SLinear(dim_latent, n_channel * 2)
        # "the biases associated with ys that we initialize to one"
        self.mapping.linear.bias.data[:n_channel] = 1
        self.mapping.linear.bias.data[n_channel:] = 0

    def forward(self, w_latent: Tensor) -> Tensor:
        # Gain scale factor and bias with:
        style = self.mapping(w_latent).unsqueeze(2).unsqueeze(3)
        return style


class AdaIn(nn.Module):
    """
    adaptive instance normalization
    """
    def __init__(self, n_channel: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(n_channel)

    def forward(self, image: Tensor, style: Tensor) -> Tensor:
        factor, bias = style.chunk(2, 1)
        result = self.norm(image)
        result = result * factor + bias
        return result


class AdaInBlock(nn.Module):
    def __init__ (self, n_channel: int, dim_latent: int):
        super().__init__()
        # Style generators
        self.style_mapping = TransformWLatentToStyle(dim_latent, n_channel)
        self.ada_in = AdaIn(n_channel)

    def forward(self, image: Tensor, latent_w: Tensor) -> Tensor:
        return self.ada_in(image, self.style_mapping(latent_w))


# Normalization on every element of input vector
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class IntermediateGenerator(nn.Module):
    '''
    A mapping consists of multiple fully connected layers.
    Used to map the input to an intermediate latent space W.
    '''

    def __init__(self, n_fc, dim_latent):
        super().__init__()
        self.mapping = nn.ModuleList()
        # layers = [PixelNorm()]
        self.mapping.append(PixelNorm())
        for i in range(n_fc):
            fc_layer = nn.Sequential(SLinear(dim_latent, dim_latent), nn.LeakyReLU(0.2))
            self.mapping.append(fc_layer)
        #     layers.append(SLinear(dim_latent, dim_latent))
        #     layers.append(nn.LeakyReLU(0.2))
        #
        # self.mapping = nn.Sequential(*layers)

    def forward(self, latent_z: Tensor) -> Tensor:
        latent_w = self.mapping(latent_z)
        return latent_w
