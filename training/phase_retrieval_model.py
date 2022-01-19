import logging
import torch
from torch import nn, Tensor
from torchvision import transforms
from typing import Dict, OrderedDict, List, Any, Optional, Union

import common.utils as utils
from common import InferredBatch, ConfigTrainer, DataBatch, S3FileSystem
from models import PhaseRetrievalPredictor, AeConv, UNetConv, BaseAe, WaveletTransformAe
from models.torch_dct import Dct2DForward
from training.utils import ModulesNames


class PhaseRetrievalAeModel:
    _log = logging.getLogger('PhaseRetrievalAeModel')

    def __init__(self, config: ConfigTrainer, log: logging.Logger, s3: S3FileSystem):
        self._config = config
        self._log = log
        self._s3 = s3
        self.n_encoder_ch = config.n_features // 2 ** (self._config.deep_ae - 1)
        if self._config.predict_out == 'features':
            # n_enc_features: int = None, n_dec_features: int = None,
            self.ae_net: Optional[BaseAe] = self._get_ae_model()
        else:
            self.ae_net: Optional[BaseAe] = None

        if self._config.use_dct_input:
            self.dct_input = Dct2DForward(utils.get_padded_size(self._config.image_size, self._config.add_pad))
        else:
            self.dct_input = None

        self._log.debug(f'=======AE-NET=======: \n {self.ae_net}')

        if self._config.predict_out == 'features':
            predict_out_ch = self.ae_net.n_enc_features_ch
            predict_out_size = self.ae_net.n_features_size
        elif self._config.predict_out == 'images':
            predict_out_ch = 1
            predict_out_size = self._config.image_size
        else:
            raise NameError(f'Nonna valid predict_out type: {self._config.predict_out}')

        self.phase_predictor = PhaseRetrievalPredictor(config=self._config,
                                                       out_ch=predict_out_ch,
                                                       out_img_size=predict_out_size)
        self._log.debug(f'=======PhaseRetrievalPredictor=======: \n {self.phase_predictor}')

        if self._config.use_ref_net:
            ch_features = self.ae_net.n_dec_features_ch if self._config.predict_out == 'features' else None
            self.ref_unet = UNetConv(n_encoder_ch=self.n_encoder_ch,
                                     deep=self._config.deep_ae,
                                     in_ch_features=ch_features,
                                     skip_input=self._config.ref_net_skip_input,
                                     active_type=self._config.activation_refnet,
                                     up_mode=self._config.up_sampling,
                                     down_pool=self._config.down_pooling_refnet,
                                     features_sigmoid_active=self._config.features_sigmoid_active,
                                     special_attention=self._config.spat_ref_net)

            self._log.debug(f'=======RefNet=======: \n {self.ref_unet}')
        else:
            self.ref_unet = None

    def _get_ae_model(self) -> BaseAe:
        if self._config.ae_type == 'conv-net':
            ae_net_model = AeConv(n_encoder_ch=self.n_encoder_ch,
                                  n_enc_features=self._config.n_features,
                                  n_dec_features=self._config.n_features_dec,
                                  img_size=self._config.image_size,
                                  deep=self._config.deep_ae,
                                  active_type=self._config.activation_ae,
                                  up_mode=self._config.up_sampling,
                                  down_pool=self._config.down_pooling_ae,
                                  features_sigmoid_active=self._config.features_sigmoid_active,
                                  use_dictionary=self._config.use_ae_dictionary,
                                  dict_len=self._config.dict_len)
        elif self._config.ae_type == 'wavelet-net':
            ae_net_model = WaveletTransformAe(img_size=self._config.image_size,
                                              deep=self._config.deep_ae,
                                              wave='db3',
                                              norm_ds=False)
        else:
            raise NameError('none valid  ae_type={self._config.ae_type}')
        return ae_net_model

    def load_module(self, state_dict: Dict[str, Tensor],
                    module: Union[nn.Module, torch.optim.Optimizer],
                    name: str,
                    force: bool = False) -> bool:
        assert isinstance(module, nn.Module) or isinstance(module, torch.optim.Optimizer)

        def is_load_module():
            return (name in state_dict) and len(self._config.load_modules) > 0 and \
                   ((self._config.load_modules[0] == 'all') or (name in self._config.load_modules))

        if is_load_module():
            self._log.info(f'Load weights of {name}')
            try:
                if isinstance(module, nn.Module):
                    missing_keys, unexpected_keys = module.load_state_dict(state_dict[name], strict=False)
                elif isinstance(module, torch.optim.Optimizer):
                    module.load_state_dict(state_dict[name])
                    missing_keys, unexpected_keys = [], []
                else:
                    raise TypeError(f'Non valid type of module {type(module)},m'
                                    f'must be nn.Module or torch.optim.Optimizer')
            except Exception as e:
                self._log.error(f'Cannot load: {name} from loaded state: {e}')
                missing_keys, unexpected_keys = [], []
                if force:
                    raise RuntimeError(f'{e}')

            if len(missing_keys):
                self._log.warning(f'{name}: Missing key  in state_dict:{missing_keys}')
            if len(unexpected_keys):
                self._log.error(f'{name}: Unexpected key  in state_dict:{unexpected_keys}')
            self._log.debug(f'Weights of {name} was been loaded')
            return True
        elif force:
            self._log.exception(f'Cannot {name} module')
            return False
        else:
            self._log.warning(f'Cannot {name} module')
            return False

    def load_modules(self, state_dict: OrderedDict[str, Tensor], force: bool = False) -> List[str]:
        loaded_models = []

        if self.ae_net and (self._config.ae_type == 'conv-net'):
            load_status = self.load_module(state_dict, self.ae_net, ModulesNames.ae_model, force)
            if load_status:
                loaded_models.append(ModulesNames.ae_model)

        if self.ref_unet:
            load_status = self.load_module(state_dict, self.ref_unet, ModulesNames.ref_net, force)
            if load_status:
                loaded_models.append(ModulesNames.ref_net)

        if self.phase_predictor:
            load_status = self.load_module(state_dict, self.phase_predictor, ModulesNames.magnitude_encoder, force)
            if load_status:
                loaded_models.append(ModulesNames.magnitude_encoder)

        return loaded_models

    def get_state_dict(self) -> Dict[str, Any]:
        save_state = {ModulesNames.config: self._config.as_dict(),
                      ModulesNames.magnitude_encoder: self.phase_predictor.state_dict()}
        if self._config.use_ref_net:
            save_state[ModulesNames.ref_net] = self.ref_unet.state_dict()

        if (self.ae_net is not None) and (self._config.ae_type == 'conv-net'):
            save_state[ModulesNames.ae_model] = self.ae_net.state_dict()

        return save_state

    def set_device(self, device: str):
        if self.ae_net is not None:
            self.ae_net.to(device=device)
        self.phase_predictor.to(device=device)
        if self.ref_unet is not None:
            self.ref_unet.to(device=device)

    def get_params(self) -> List[Tensor]:
        gen_params = list(self.phase_predictor.parameters())
        if self._config.use_ref_net:
            gen_params += list(self.ref_unet.parameters())
        return gen_params

    def forward_magnitude_encoder(self, data_batch: DataBatch) -> InferredBatch:
        if (self._config.gauss_noise is not None) and self._config.use_aug:
            fft_magnitude = data_batch.fft_magnitude_noised
        else:
            fft_magnitude = data_batch.fft_magnitude
        enc_features_batch_recon, intermediate_features = self.phase_predictor(fft_magnitude)
        if self._config.predict_out == 'features':
            dec_features_batch_recon, coeff_recon = self.ae_net.map_to_dec_features(enc_features_batch_recon)
            recon_batch = self.ae_net.decode(dec_features_batch_recon)
        elif self._config.predict_out == 'images':
            recon_batch = enc_features_batch_recon

        if self._config.predict_out == 'features':
            feature_encoder = self.ae_net.encode(data_batch.image)
            feature_decoder, coeff_enc = self.ae_net.map_to_dec_features(feature_encoder)
            decoded_batch = self.ae_net.decode(feature_decoder)
        else:
            feature_encoder = None
            decoded_batch = None
            dec_features_batch_recon = None

        inferred_batch = InferredBatch(img_recon=recon_batch,
                                       feature_recon=enc_features_batch_recon,
                                       feature_recon_decoder=dec_features_batch_recon,
                                       feature_encoder=feature_encoder,
                                       feature_decoder=feature_decoder,
                                       decoded_img=decoded_batch,
                                       intermediate_features=intermediate_features,
                                       dict_coeff_encoder=coeff_enc,
                                       dict_coeff_recon=coeff_recon)

        if self._config.use_ref_net:
            inferred_batch.img_recon_ref = self.ref_unet(recon_batch.detach(), dec_features_batch_recon.detach())
            inferred_batch.fft_magnitude_recon_ref = self.forward_magnitude_fft(inferred_batch.img_recon_ref)

        return inferred_batch

    def forward_ae(self, data_batch: DataBatch) -> InferredBatch:
        if (self._config.gauss_noise is not None) and self._config.use_aug:
            img_batch = data_batch.image_noised
        else:
            img_batch = data_batch.image
        recon_batch, enc_features_batch, dec_features_batch, coeff_enc = self.ae_net(img_batch)
        enc_feature_recon = self.ae_net.encode(recon_batch)
        return InferredBatch(img_recon=recon_batch,
                             feature_encoder=enc_features_batch,
                             feature_recon=enc_feature_recon,
                             feature_recon_decoder=dec_features_batch,
                             feature_decoder=dec_features_batch,
                             dict_coeff_encoder=coeff_enc)

    def forward_magnitude_fft(self, data_batch: Tensor) -> Tensor:
        if self._config.add_pad > 0.0:
            pad_value = utils.get_pad_val(self._config.image_size, self._config.add_pad)
            data_batch_pad = transforms.functional.pad(data_batch, pad_value, padding_mode='edge')
        else:
            data_batch_pad = data_batch
        if self._config.use_dct_input:
            magnitude_batch = self.dct_input(data_batch_pad).abs()
        elif self._config.use_rfft:
            magnitude_batch = torch.fft.rfft2(data_batch_pad, norm=self._config.fft_norm).abs()
        else:
            magnitude_batch = torch.fft.fft2(data_batch_pad, norm=self._config.fft_norm).abs()
        return self._config.spectral_factor * magnitude_batch

    def set_eval_mode(self):
        self._log.debug('Set model  eval mode')
        self.phase_predictor.eval()
        if self.ae_net is not None:
            self.ae_net.eval()
        if self._config.use_ref_net:
            self.ref_unet.eval()

    def set_train_mode(self, ae_train: bool = False):
        self._log.debug(f'Set model  train mode: ae_train: {ae_train}')
        if self.phase_predictor:
            self.phase_predictor.train()
        self.ae_net.eval()
        # if self.ae_net:
        #     if ae_train:
        #         self.ae_net.train()
        #     else:
        #         self.ae_net.eval()
        if self._config.use_ref_net:
            self.ref_unet.train()
