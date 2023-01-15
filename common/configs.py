import json
import warnings

import yaml
import jsonpickle
import numpy as np
import os
from typing import Optional, List, Iterable, Sequence, Union, Tuple
from dataclasses import dataclass, field
from .paths import PATHS
from .aws_utils import S3FileSystem


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
    def from_yaml(cls, input_path: str) -> 'ConfigBase':
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
            warnings.warn(f'Not valid name: {non_valid_keys}, will be removed from config dict')
            [obj_dict.pop(key_to_remove, None) for key_to_remove in non_valid_keys]
            
        obj = cls.from_dict(obj_dict)

        return obj

    @classmethod
    def from_data_file(cls, input_path) -> 'Config':
        if input_path.endswith('.json'):
            return cls.from_json(input_path)
        elif input_path.endswith('.yaml'):
            return cls.from_yaml(input_path)
        else:
            raise ValueError(f'Non-supported format of file: {input_path}. '
                             f'Supported formats are JSON or YAML.')

    @classmethod
    def load_from_path(cls, config_path: str) -> 'ConfigBase':
        s3 = S3FileSystem()
        if s3.is_s3_url(config_path):
            assert s3.isfile(config_path)
            config = s3.load_object(config_path, cls.from_data_file)
        else:
            assert os.path.exists(config_path)
            config = cls.from_data_file(config_path)
        return config

    @classmethod
    def load_config(cls, config_obj: Union[str, dict], **kwargs) -> 'Config':
        if config_obj is None:
            config = cls()
        elif isinstance(config_obj, str):
            config = cls.load_from_path(config_obj)
        elif isinstance(config_obj, dict):
            config = cls.from_dict(config_obj)
        config.log_path = PATHS.LOG
        config = config.update(**kwargs)
        return config

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
class ConfigSpikesTrainer(ConfigBase):
    # Base
    project_name = 'phase-spikes-retrieval'
    task_name: str = 'sparse-spikes-prediction'
    dataset_name: str = 'spikes-generator'
    seed: int = 1
    use_tensor_board: bool = True
    debug_mode: bool = False

    # training
    batch_size_train: int = 16
    batch_size_test: int = 32
    learning_rate = 0.0001
    lr_milestones: List[int] = field(default_factory=lambda: [15, 20, 25, 30, 32])
    n_epochs_pr: int = 35
    n_iter_tr: int = 1000
    cuda: bool = True
    n_dataloader_workers: int = 8  # n_dataloader_workers

    # eval
    n_iter_eval: int = 500
    save_model_interval: int = 10

    # logger
    log_interval: int = 500
    log_image_interval: int = 5000
    log_eval_interval: int = 5000
    dbg_img_batch: int = 8

    # model
    fft_norm: str = "ortho"
    model_type: str = 'mlp'
    proj_mag: bool = False
    use_dct_input: bool = False
    use_noised_input: bool = True
    predict_out: str = 'images'
    path_pretrained: Optional[str] = None
    count_predictor_head: bool = False
    conv_net_deep: int = 3
    n_encoder_ch: int = 16
    multi_scale_out: bool = False

    # optimization
    loss_type_img_recon: str = 'l1'
    loss_type_mag: str = 'l2'
    lambda_support_size: float = 1.0  # 0.01
    lambda_img_recon: float = 20.0  # 2.0
    lambda_fft_recon: float = 2.0  # 20.0
    lambda_count_spikes: float = 2.0
    lambda_sparsity: float = 0.1

    # data
    image_size: int = 32
    tile_size: int = 32
    pad: int = 0
    spikes_range: Union[int, Tuple[int, int]] = 5
    gauss_noise: float = 0.005
    sigma: float = 0.75
    shift_fft: bool = False
    use_aug: bool = False
    use_rfft: bool = False


@dataclass
class ConfigTrainer(ConfigBase):
    project_name = 'phase-retrieval'
    task_name: str = 'ae-features-prediction'
    dataset_name: str = 'mnist'
    predict_out: str = 'features' # 'images'
    use_tensor_board: bool = True
    seed: int = 1
    use_ref_net: bool = False
    log_path: Optional[str] = PATHS.LOG
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
    predict_type: str = 'spectral'  # special, phase
    predict_fc_multy_coeff: float = 2.0
    predict_conv_type: str = 'ConvBlock' # ResBlock, Unet
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
    lambda_sparsity_dict_coeff: float = 1.0
    lambda_discrim_img: float = 1.0
    lambda_discrim_ref_img: float = 1.0
    lambda_discrim_features: float = 1.0
    lambda_fc_pred_l1_req: float = 0.0
    lambda_fc_layers_pred_l1_req: List[float] = field(default_factory=lambda:  [1.0, 1.0, 1.0])
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
    affine_aug: bool = False
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
    prob_aug: float = 0.25
    gamma_corr: Optional[Sequence[float]] = None # for face images  field(default_factory=lambda:  (0.85, 1.125))
    gauss_blur: Optional[Sequence[float]] = None # for face images field(default_factory=lambda:  (0.5, 1.5))
    sharpness_factor: Optional[float] = None
    rnd_vert_flip: bool = False
    rnd_horiz_flip: bool = False  # for face images
    gauss_noise: Optional[Union[float, Sequence[float]]] = None
    add_pad: float = 0.0
    rot_degrees: Optional[Sequence[float]] = field(default_factory=lambda:  [2.0, 5.0])
    translation: Optional[Sequence[float]] = field(default_factory=lambda:  [0.025, 0.025])
    scale: Optional[Sequence[float]] = field(default_factory=lambda:  [0.9, 1.2])
    use_dropout_enc_fc: bool = False
    use_norm_enc_fc: Optional[str] = None
    activation_fc_enc: str = 'leakly_relu'
    activation_fc_ch_params: bool = False
    use_dct: bool = False
    use_dct_input: bool = False
    norm_fc: str = None  # None: no, 'batch_norm', 'layer_norm', 'instance_norm'
    magnitude_norm: str = None  # None: no, 'batch_norm', 'layer_norm', 'instance_norm'
    inter_norm: str = None # None: no, 'batch_norm', 'layer_norm', 'instance_norm'
    spectral_factor: float = 1.0
    use_ae_dictionary: bool = False
    add_pad_out: float = 0.0
    dict_len: int = 16
    n_features_dec: int = 64
    save_model_interval: int = 10
    spat_ref_net:  bool = False
    lr_ae: Optional[float] = None
    lr_ref_net: Optional[float] = None
    lr_enc: Optional[float] = None
    lr_dict: Optional[float] = None
    lr_discr: Optional[float] = None
    lr_ae_decoder: Optional[float] = None
    optim_exclude:  List[str] = field(default_factory=lambda: [])
    spat_conv_predict: bool = False
    ae_decoder_fine_tune: bool = False
    lambda_img_ae_loss_l2: float = 1.0
    lambda_img_ae_loss_l1: float = 1.0
    ae_type: str = 'conv-net'  # 'conv-net', 'wavelet-net'
    wavelet_type: Optional[str] = None  # 'haar' 'db3'
    use_conv_block_predictor: bool = True
    ref_unet_in_features: bool = True
    deep_ref_unet: Optional[int] = None
    image_crop_size: Optional[int] = None
    conv_pred_norm: Optional[str] = 'batch_norm'




