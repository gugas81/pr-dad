import torch
from torch import Tensor
import numpy as np
import dataclasses
from typing import Optional, List, Callable, Dict, Any
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class DataBatch:
    def __len__(self):
        v = dataclasses.astuple(self)[0]
        return v.size()[0]

    @staticmethod
    def _merge(params: List['DataBatch'], merge_func: Callable):
        new_obj = defaultdict(list)

        for p in params:
            for key, value in dataclasses.asdict(p).items():
                if value is not None and new_obj[key] is not None:
                    new_obj[key].append(value)
                else:
                    new_obj[key] = None

        new_object_by_type = {}
        for key, value in new_obj.items():
            if value is None:
                new_object_by_type[key] = None
                continue

            new_object_by_type[key] = merge_func(new_obj[key])

        return params[0].__class__(**new_object_by_type)

    def apply(self, apply_func: Callable):
        torch_data = {}
        for key, val in self.__dict__.items():
            if val is not None:
                torch_data[key] = apply_func(val)
            else:
                torch_data[key] = None
        return self.__class__(**torch_data)

    def reduce(self, merge_func: Callable):
        torch_data = {}
        for key, val in self.__dict__.items():
            if val is not None:
                torch_data[key] = merge_func(val)
            else:
                torch_data[key] = None

        return self.__class__(**torch_data)

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def get_keys(self) -> List[str]:
        return list(self.__dict__.keys())

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DataBatch':
        return cls(**d)


@dataclass
class NumpyBatch(DataBatch):

    @staticmethod
    def merge(params: List['NumpyBatch']) -> 'NumpyBatch':
        return DataBatch._merge(params, np.array)

    def mean(self):
        return self.reduce(np.mean)

    def std(self):
        return self.reduce(np.std)

    def min(self):
        return self.reduce(np.min)

    def max(self):
        return self.reduce(np.max)


@dataclass
class TensorBatch(DataBatch):

    @staticmethod
    def merge(params: List['TensorBatch']):
        return TensorBatch._merge(params, torch.stack)

    def to(self, device: str):
        torch_data = {}
        for key, val in dataclasses.asdict(self).items():
            if val is not None:
                if isinstance(val, Tensor):
                    torch_data[key] = val.to(device=device)
                elif isinstance(val, dict):
                    torch_data[key] = {key_: val_.to(device=device) for key_, val_ in val.items()}
                else:
                    raise ValueError(f'Key: {key}, type={type(val)}')
            else:
                torch_data[key] = val
        return self.__class__(**torch_data)

    def mean(self):
        return self.reduce(torch.mean)

    def detach(self):
        return self.reduce(torch.detach)


@dataclass
class DataBatch(TensorBatch):
    image: Tensor = torch.tensor([float('nan')])
    image_noised = torch.tensor([float('nan')])
    image_discrim: Tensor = torch.tensor([float('nan')])
    fft_magnitude: Tensor = torch.tensor([float('nan')])
    fft_magnitude_noised: Tensor = torch.tensor([float('nan')])
    label: Tensor = torch.tensor([float('nan')])
    label_discrim: Tensor = torch.tensor([float('nan')])
    is_paired: bool = True


@dataclass
class InferredBatch(TensorBatch):
    img_recon: Optional[Tensor] = None
    img_recon_ref: Optional[Tensor] = None
    fft_magnitude_recon_ref: Optional[Tensor] = None
    feature_recon: Optional[Tensor] = None
    feature_recon_decoder: Optional[Tensor] = None
    feature_encoder: Optional[Tensor] = None
    feature_decoder: Optional[Tensor] = None
    decoded_img: Optional[Tensor] = None
    intermediate_features: Optional[Tensor] = None
    dict_coeff_encoder: Optional[Tensor] = None
    dict_coeff_recon: Optional[Tensor] = None


@dataclass
class DiscriminatorBatch(TensorBatch):
    validity:  Optional[Tensor] = None
    features: Optional[List[Tensor]] = None


@dataclass
class Losses(TensorBatch):
    def __str__(self) -> str:
        losses_batch_str = [f'{metric_name}: {val_losses.mean().detach().cpu().numpy(): .4f}'
                            for metric_name, val_losses in self.__dict__.items() if val_losses is not None]
        losses_batch_str = ' '.join(losses_batch_str)
        return losses_batch_str


@dataclass
class LossesPRImages(Losses):
    l1norm_real_part: Tensor
    l2norm_imag_part: Tensor
    mse_magnitude: Tensor
    mse_img: Tensor
    l1norm_real_part_ref: Optional[Tensor] = None
    l1_orig: Optional[Tensor] = None
    mse_img_ref: Optional[Tensor] = None
    mse_magnitude_ref: Optional[Tensor] = None
    total: Optional[Tensor] = None


@dataclass
class LossesPRFeatures(Losses):
    l1_img: Optional[Tensor] = None
    l2_img: Optional[Tensor] = None
    l2_img_np: Optional[Tensor] = None
    l1_ref_img: Optional[Tensor] = None
    l2_ref_img: Optional[Tensor] = None
    l2_ref_img_np: Optional[Tensor] = None
    lpips_img: Optional[Tensor] = None
    lpips_ref_img: Optional[Tensor] = None
    l1_features: Optional[Tensor] = None
    l2_features: Optional[Tensor] = None
    l1_magnitude: Optional[Tensor] = None
    l2_magnitude: Optional[Tensor] = None
    l1_ref_magnitude: Optional[Tensor] = None
    l2_ref_magnitude: Optional[Tensor] = None
    l1_sparsity_dict_coeff: Optional[Tensor] = None
    realness_features: Optional[Tensor] = None
    img_adv_loss: Optional[Tensor] = None
    ref_img_adv_loss: Optional[Tensor] = None
    features_adv_loss: Optional[Tensor] = None
    img_disrm_loss: Optional[Tensor] = None
    ref_img_disrm_loss: Optional[Tensor] = None
    features_disrm_loss: Optional[Tensor] = None
    disrm_loss: Optional[Tensor] = None
    l1_sparsity_features: Optional[Tensor] = None
    perceptual_disrim_img: Optional[Tensor] = None
    perceptual_disrim_ref_img: Optional[Tensor] = None
    perceptual_disrim_features: Optional[Tensor] = None
    lr:  Optional[Dict[str,Tensor]] = None
    mean_img: Optional[Tensor] = None
    std_img: Optional[Tensor] = None
    max_img: Optional[Tensor] = None
    min_img: Optional[Tensor] = None

    l1_reg_fc_pred: Optional[Tensor] = None

    mean_features: Optional[Tensor] = None
    std_features: Optional[Tensor] = None
    max_features: Optional[Tensor] = None
    min_features: Optional[Tensor] = None

    mean_img_ref: Optional[Tensor] = None
    std_img_ref: Optional[Tensor] = None
    max_img_ref: Optional[Tensor] = None
    min_img_ref: Optional[Tensor] = None

    total: Optional[Tensor] = None


@dataclass
class LossesGradNorms(Losses):
    l2_grad_img_discriminator:  Optional[Tensor] = None
    l2_grad_features_discriminator: Optional[Tensor] = None
    l2_grad_magnitude_encoder: Optional[Tensor] = None





