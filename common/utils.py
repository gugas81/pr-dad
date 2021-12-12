import numpy as np
import random
import torch
from torch import Tensor
from torch.nn import functional as F
from typing import List
import torchvision
from lpips import LPIPS


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


class LossImg(torch.nn.Module):
    def __init__(self, loss_type: str = 'l2', rot180: bool = False, device: str = None, eval_mode: bool = True):
        super(LossImg, self).__init__()
        self._rot180 = rot180
        self._loss_type = loss_type
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


def get_fft2_freq(img_size: int, use_rfft: bool) -> List[int]:
    if use_rfft:
        fft_freq_1d = torch.fft.rfftfreq(img_size)
    else:
        fft_freq_1d = torch.fft.fftfreq(img_size)
    return (fft_freq_1d * img_size).numpy().astype(int).tolist()


def get_pad_val(img_size: int, add_pad: float) -> int:
    return int(0.5 * add_pad * img_size)


def get_padded_size(img_size: int, add_pad: float) -> int:
    pad_val = get_pad_val(img_size, add_pad)
    return 2 * pad_val + img_size


def get_fft2_last_size(img_size: int, use_rfft: bool) -> int:
    return len(get_fft2_freq(img_size, use_rfft))


def get_magnitude_size_2d(img_size: int, pad_val: int, use_rfft: bool) -> List[int]:
    pad_size = get_padded_size(img_size, pad_val)
    return [pad_size, get_fft2_last_size(pad_size, use_rfft)]


def get_flatten_fft2_size(img_size:  int, use_rfft: bool) -> int:
    return img_size * get_fft2_last_size(img_size, use_rfft)

