import numpy as np
import random
import torch
from torch import Tensor
from torch.nn import functional as F
from typing import List
import torchvision


def set_seed(seed: int):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def l2_grad_norm(model: torch.nn.Module) -> (Tensor, Tensor):
    norm_params = [p.grad.data.norm(2) if p.grad is not None else torch.tensor(0.0, device=p.device)
                   for p in model.parameters()]
    norm_params = torch.stack(norm_params)
    total_norm = torch.sqrt(torch.sum(torch.square(norm_params)))
    return total_norm, norm_params


def l2_perceptual_loss(x: List[Tensor], y: List[Tensor], weights: List[float]) -> Tensor:
    loss_out = torch.zeros([], device=x[0].device, dtype=x[0].dtype)
    for x_layer, y_layer, w in zip(x, y, weights):
        loss_out += w * F.mse_loss(x_layer, y_layer)

    return loss_out


class LossImg:
    def __init__(self, loss_type: str = 'l2', rot180: bool = False):
        self._rot180 = rot180
        if self._rot180:
            reduction = 'none'
        else:
            reduction = 'mean'

        if loss_type == 'l2':
            self._loss_fun = torch.nn.MSELoss(reduction=reduction)
        elif loss_type == 'l1':
            self._loss_fun = torch.nn.L1Loss(reduction=reduction)
        else:
            raise TypeError(f'Non valid loss type: {loss_type}')

    def forward(self, in_img: Tensor, tgt_img: Tensor) -> Tensor:
        if self._rot180:
            in_img_rot180 = torch.rot90(in_img, 2, (-2, -1))
            img_rot180_2d = torch.cat((in_img, in_img_rot180), -1)
            tgt_img_2d = torch.unsqueeze(tgt_img, -1)
            loss = self._loss_fun(img_rot180_2d, tgt_img_2d)
            loss = loss.mean((-3, -2))
            loss, _ = loss.min(dim=-1)
            loss = loss.mean()
        else:
            loss = self._loss_fun(in_img, tgt_img)
        return loss


class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return super().__call__(tensor.clone())


