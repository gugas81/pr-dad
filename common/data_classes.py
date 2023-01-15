import torch
from torch import Tensor
import numbers
import numpy as np
import dataclasses
from typing import Optional, List, Callable, Dict, Any
from collections import defaultdict
from dataclasses import dataclass, field
from collections.abc import Iterable  # import directly from collections for Python < 3.3


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
                    if isinstance(value, dict):
                        if key not in new_obj or len(new_obj[key]) == 0:
                            new_obj[key] = defaultdict(list)
                        for key_, val_ in value.items():
                            new_obj[key][key_].append(val_)
                    else:
                        new_obj[key].append(value)
                else:
                    new_obj[key] = None

        new_object_by_type = {}
        for key, value in new_obj.items():
            if value is None:
                new_object_by_type[key] = None
                continue
            if isinstance(value, dict) or isinstance(value, defaultdict):
                sub_obj = {key_: merge_func(val_) for key_, val_ in value.items()}
                new_object_by_type[key] = sub_obj
            else:
                new_object_by_type[key] = merge_func(value)

        return params[0].__class__(**new_object_by_type)

    def apply(self, apply_func: Callable):
        torch_data = {}
        apply_f = lambda x: apply_func(x) if isinstance(x, Tensor) or isinstance(x, np.ndarray) or isinstance(x, numbers.Number) else x
        for key, val in self.__dict__.items():
            if isinstance(val, dict):
                torch_data[key] = {key_: apply_f(val_) for key_, val_ in val.items()}
            else:
                torch_data[key] = apply_f(val)
        return self.__class__(**torch_data)

    def reduce(self, merge_func: Callable):
        merge_f = lambda x: merge_func(x) if isinstance(x, Tensor) or isinstance(x, np.ndarray) or isinstance(x, numbers.Number) else x
        torch_data = {}
        for key, val in self.__dict__.items():
            if isinstance(val, dict):
                torch_data[key] = {key_: merge_f(val_) for key_, val_ in val.items()}
            else:
                torch_data[key] = merge_f(val)

        return self.__class__(**torch_data)

    def get_subset(self, index):
        dict_data = {}
        for key, val in self.__dict__.items():
            if isinstance(val, Iterable):
                val = val[:index]
            dict_data[key] = val
        return self.__class__(**dict_data)

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

    def to(self, device: str) -> 'TensorBatch':
        def to_device(val_x: Any) -> Any:
            return val_x.to(device=device) if isinstance(val_x, Tensor) else val_x

        torch_data = {}
        for key, val in dataclasses.asdict(self).items():
            if val is not None:
                if isinstance(val, dict):
                    torch_data[key] = {key_: to_device(val_) for key_, val_ in val.items()}
                else:
                    torch_data[key] = to_device(val)
            else:
                torch_data[key] = None
        return self.__class__(**torch_data)

    def mean(self):
        return self.reduce(torch.mean)

    def detach(self):
        return self.reduce(torch.detach)

    def to_numpy_dict(self) -> dict:
        numpy_f = lambda x: torch.Tensor.numpy(x, force=True) if isinstance(x, Tensor) else x
        numpy_data = {}
        for key, val in self.__dict__.items():
            if isinstance(val, dict):
                numpy_data[key] = {key_: numpy_f(val_) for key_, val_ in val.items()}
            else:
                numpy_data[key] = numpy_f(val)

        return numpy_data


@dataclass
class DataSpikesBatch(TensorBatch):
    image: Tensor = torch.tensor([float('nan')])
    image_noised: Tensor = torch.tensor([float('nan')])
    fft_magnitude: Tensor = torch.tensor([float('nan')])
    fft_magnitude_noised: Tensor = torch.tensor([float('nan')])
    n_spikes: int = torch.tensor([float('nan')])
    x: Tensor = torch.tensor([float('nan')])
    y: Tensor = torch.tensor([float('nan')])


@dataclass
class InferredSpikesBatch(TensorBatch):
    img_recon: Optional[Tensor] = None
    img_recon_scales: Optional[List[Tensor]] = None
    fft_recon: Optional[Tensor] = None
    pred_n_spikes: Optional[Tensor] = None
    orig_blur_imgs: Optional[List[Tensor]] = None
    recon_blur_imgs: Optional[List[Tensor]] = None


@dataclass
class DataBatch(TensorBatch):
    image: Tensor = torch.tensor([float('nan')])
    image_noised: Tensor = torch.tensor([float('nan')])
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
        def get_repr(val) -> str:
            get_str = lambda v_: f'{v_.mean().detach().cpu().numpy(): .4f}' if isinstance(v_, Tensor) else str(v_)
            if isinstance(val, dict):
                out_str = [f'{key}:{get_str(val_)}' for key, val_ in val.items()]
                out_str = ' '.join(out_str)
            else:
                out_str = get_str(val)
            return out_str
        losses_batch_str = [f'{metric_name}: {get_repr(val_losses)}'
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
    l1_ae_img: Optional[Tensor] = None
    l2_ae_img: Optional[Tensor] = None
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
    lr:  Optional[Dict[str, Tensor]] = None
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


@dataclass
class LossesSpikesImages(Losses):
    img_recon: Optional[Tensor] = None
    img_scale_recon: Optional[Dict[str, Tensor]] = None
    img_sparsity: Optional[Tensor] = None
    fft_recon: Optional[Tensor] = None
    count_pred_loss: Optional[Tensor] = None
    support_size: Optional[Tensor] = None
    total: Optional[Tensor] = None
    lr: Optional[Dict[str, Tensor]] = None


