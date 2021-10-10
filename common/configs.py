import json
import yaml
import jsonpickle
import numpy as np
from typing import Optional, List, Callable, Iterable, Dict
from dataclasses import dataclass, field
from .paths import PATHS


class NumpyEncoder(json.JSONEncoder):
    """
    A JSONEncoder capable of converting numpy types to simple python builtin types.
    """

    # pylint: disable=method-hidden
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()

        if isinstance(o, (np.float16, np.float32, np.float64)):
            return float(o)

        if isinstance(o, (np.int64, np.int32, np.int16)):
            return int(o)

        return json.JSONEncoder.default(self, o)


@dataclass
class ConfigBase:
    def __str__(self):
        s = f'\n[{self.__class__.__name__} -----'
        for key, value in sorted(self.__dict__.items()):
            s += f'\n o {key:45} | {value}'
        s += f'\n ----- {self.__class__.__name__}]'
        return s

    def as_dict(self):
        return self.__dict__.copy()

    def __getitem__(self, key):
        return self.__dict__[key]

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    @classmethod
    def from_yaml(cls, input_path: str):
        with open(input_path, 'r') as f:
            obj: ConfigBase = yaml.load(f)

        objs = obj if isinstance(obj, Iterable) else [obj]
        return objs

    def to_yaml(self, output_path: str):
        with open(output_path, 'w') as f:
            yaml.dump(self, f)

    def to_json(self, output_path: str, indent: int = 4, encoder: json.JSONEncoder = NumpyEncoder):
        json_str = jsonpickle.encode(self)
        # load the encoded JSON so we can save it with in a pretty
        # format with indentations (jsonpickle does not support this)
        json_dict = json.loads(json_str)
        with open(output_path, 'w') as f:
            json.dump(json_dict, f, indent=indent, cls=encoder)

    @classmethod
    def get_fields_names(cls) -> List[str]:
        return list(cls.__dataclass_fields__.keys())

    @classmethod
    def from_json(cls, input_path: str) -> 'Config':
        with open(input_path, 'r') as f:
            json_obj = jsonpickle.decode(f.read())
        if isinstance(json_obj, dict):
            obj_dict = json_obj
        else:
            assert isinstance(json_obj, ConfigBase)
            obj_dict = json_obj.__dict_
        non_valid_keys = [key for key in obj_dict.keys() if key not in cls.get_fields_names()]
        if len(non_valid_keys) > 0:
            raise NameError(f'Not valid name: {non_valid_keys}')
        obj = cls.from_dict(obj_dict)

        return obj

    @classmethod
    def from_data_file(cls, input_path):
        if input_path.endswith('.json'):
            return cls.from_json(input_path)
        elif input_path.endswith('.yaml'):
            return cls.from_yaml(input_path)
        else:
            raise ValueError(f'Non-supported format of file: {input_path}. '
                             f'Supported formats are JSON or YAML.')

    def update(self, **kwargs) -> 'Config':
        """
        Updates the values of given configuration elements
        :param kwargs: The names and values of the configuration elements to update.
        :return: self
        """

        try:
            for key, value in kwargs.items():
                c = self.__dict__
                *sub_keys, final_key = key.split('.')
                for sub_key in sub_keys:
                    c = c[sub_key].__dict__
                c[final_key] = value
        except KeyError as e:
            raise KeyError(f'Non-valid update input {kwargs} , error message: {e}')

        return self


@dataclass
class ConfigTrainer(ConfigBase):
    project_name = 'phase-retrieval'
    task_name: str = 'ae-features-prediction'
    dataset_name: str = 'mnist'
    predict_out: str = 'features' # 'images'
    use_tensor_board: bool = True
    seed: int = 1
    use_ref_net: bool = False
    log_path: Optional[str] = field(default_factory=PATHS.LOG)
    n_epochs_pr: int = 50
    lr_milestones_en: List[int] = field(default_factory=lambda: [20, 30, 40])
    lr_reduce_rate_en: float = 0.5
    batch_size_train: int = 64
    batch_size_test: int = 1000
    n_features: int = 64
    n_epochs_ae: int = 10
    lr_milestones_ae: List[int] = field(default_factory=lambda: [5, 7])
    lr_reduce_rate_ae: float = 0.5
    path_pretrained: Optional[str] = None
    load_modules: List[str] = field(default_factory=lambda: ['all'])
    learning_rate: float = 0.0001
    is_train_ae: bool = True
    is_train_encoder: bool = True
    use_ref_net: bool = False
    use_gan: bool = True
    dbg_img_batch: int = 4
    dbg_features_batch: int = 4
    log_interval: int = 100
    log_image_interval: int = 500
    fft_norm: str = "ortho"
    predict_type: str = 'spectral'
    predict_fc_multy_coeff: int = 2
    predict_conv_type: str = 'ConvBlock'
    seed: int = 1
    use_aug: bool = True
    debug_mode: bool = False
    cuda: bool = True
    lambda_img_recon_loss: float = 1.0
    lambda_img_recon_loss_l1: float = 1.0
    lambda_ref_img_recon_loss: float = 1.0
    lambda_ref_img_recon_l1: float = 1.0
    lambda_img_adv_loss: float = 1.0
    lambda_img_perceptual_loss: float = 1.0
    lambda_ref_img_adv_loss: float = 1.0
    lambda_ref_img_perceptual_loss: float = 1.0
    lambda_features_adv_loss: float = 1.0
    lambda_features_recon_loss: float = 1.0
    lambda_features_recon_loss_l1: float = 1.0
    lambda_features_perceptual_loss: float = 1.0
    lambda_magnitude_recon_loss_l1: float = 1.0
    lambda_ref_magnitude_recon_loss_l1: float = 1.0
    lambda_magnitude_recon_loss: float = 1.0
    lambda_ref_magnitude_recon_loss: float = 1.0
    lambda_features_realness: float = 1.0
    lambda_sparsity_features: float = 1.0
    lambda_discrim_img: float = 1.0
    lambda_discrim_ref_img: float = 1.0
    lambda_discrim_features: float = 1.0
    clip_encoder_grad: Optional[float] = 0.5
    clip_discriminator_grad: Optional[float] = 0.1
    disrim_in_conv_ch: Optional[int] = None
    disrim_input_norm: Optional[str] = None
    disrim_fc_layers: List[int] = field(default_factory=lambda:  [1024, 512, 128])
    disrim_features_fc_layers: List[int] = field(default_factory=lambda: [2048, 1024, 512, 128, 64])
    disrim_fc_norm: Optional[str] = None
    image_size: int = 32
    n_dataloader_workers: int = 1
    deep_ae: int = 3
    part_supervised_pairs: float = 1.0
    ref_net_skip_input: bool = False
    weights_plos: List[float] = field(default_factory=lambda:  [1.0, 0.0])
    features_sigmoid_active: bool = False
    down_pooling_ae: str = 'avrg_pool'  # avrg_pool max_pool
    down_pooling_refnet: str = 'avrg_pool'  # avrg_pool max_pool
    activation_ae: str = 'leakly_relu' # relu leakly_relu
    activation_refnet: str = 'leakly_relu'  # relu leakly_relu
    activation_enc: str = 'leakly_relu'  # relu leakly_relu
    activation_discrim: str = 'leakly_relu'  # relu leakly_relu
    up_sampling: str = 'bilinear' # 'bilinear' 'nearest'
    rot_degrees: float = 0.0
    use_aug_test: bool = False
    use_rfft: bool = False
    loss_rot180: bool = False
    n_inter_features: Optional[int] = None
    discrim_features_ch: Optional[int] = None
    deep_predict_fc: Optional[int] = None
    deep_predict_conv: int = 2
    use_amp: bool = False
    predict_conv_multy_coeff: int = 2
    predict_img_int_features_multi_coeff: int = 1
    use_lpips: bool = False
    lambda_img_lpips: float = 1.0
    lambda_ref_img_lpips: float = 1.0
    use_adain: bool = False
