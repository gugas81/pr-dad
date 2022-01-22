import numpy as np
import random
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import List, Optional
from lpips import LPIPS


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


class LossImg(nn.Module):
    def __init__(self, loss_type: str = 'l2', rot180: bool = False, device: str = None, eval_mode: bool = True):
        super(LossImg, self).__init__()
        self._rot180 = rot180
        self._loss_type = loss_type
        self._device = device
        if self._rot180:
            self._reduction = 'none'
        else:
            self._reduction = 'mean'

        if self._loss_type == 'l2':
            self._loss_fun = torch.nn.MSELoss(reduction=self._reduction)
        elif self._loss_type == 'l1':
            self._loss_fun = torch.nn.L1Loss(reduction=self._reduction)
        elif self._loss_type == 'lpips':
            self._loss_fun = LPIPS(net='vgg', eval_mode=eval_mode, verbose=False)
            self._reduction = 'none'
        else:
            raise TypeError(f'Non valid loss type: {self._loss_type}')

        if device is not None:
            self._loss_fun.to(device=device)

    def forward(self, in_img: Tensor, tgt_img: Tensor) -> Tensor:
        if self._rot180:
            in_img_rot180 = torch.rot90(in_img, 2, (-2, -1))
            loss = self._loss_fun(in_img, tgt_img).mean((-3, -2, -1))
            loss_rot = self._loss_fun(in_img_rot180, tgt_img).mean((-3, -2, -1))
            loss = torch.min(loss, loss_rot)
        else:
            loss = self._loss_fun(in_img, tgt_img)
        if self._reduction == 'none':
            loss = loss.mean()
        return loss


class DwtCoeffLoss(LossImg):
    def __init__(self,
                 n_subbands: int,
                 loss_type: str = 'l2',
                 rot180: bool = False,
                 device: str = None,
                 eval_mode: bool = True,
                 w_dc: float = 0.1,
                 lambda_shrink: Optional[float] = None,
                 ):
        super(DwtCoeffLoss, self).__init__(loss_type=loss_type, rot180=rot180, device=device, eval_mode=eval_mode)
        self._w_dc = w_dc
        self._lambda_shrink = lambda_shrink
        self._w_subbands = torch.cat([self._w_dc * torch.ones((1, 1, 1, 1), device=self._device),
                                      torch.ones((1, n_subbands - 1, 1, 1), device=self._device)], dim=1)

    def forward(self, in_coeff: Tensor, tgt_coeff: Tensor) -> Tensor:
        return super().forward(self._w_subbands * in_coeff, self._w_subbands * tgt_coeff)


class SparsityL1Loss(nn.Module):
    def __init__(self, dc_comp: bool = False):
        super(SparsityL1Loss, self).__init__()
        self._dc_comp = dc_comp

    def forward(self, x: Tensor) -> Tensor:
        if self._dc_comp:
            sparsity = torch.mean(torch.index_select(x, 1, torch.arange(1, x.shape[1], device=x.device)).abs())
        else:
            sparsity = torch.mean(x.abs())
        return sparsity
