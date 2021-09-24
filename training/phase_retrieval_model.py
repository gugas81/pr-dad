import logging
from typing import Dict, OrderedDict, List, Any
import torch
from torch import nn
from torch import Tensor

from common import InferredBatch, ConfigTrainer
from common import DataBatch, S3FileSystem

from training.utils import ModulesNames
from models import PhaseRetrievalPredictor,   AeConv, UNetConv


class PhaseRetrievalAeModel:

    _log = logging.getLogger('PhaseRetrievalAeModel')

    def __init__(self, config: ConfigTrainer, log: logging.Logger, s3: S3FileSystem):
        self._config = config
        self._log = log
        self._s3 = s3
        self.n_encoder_ch = config.n_features // 2**(self._config.deep_ae-1)
        if self._config.predict_out == 'features':
            self.ae_net = AeConv(n_encoder_ch=self.n_encoder_ch,
                                 img_size=self._config.image_size,
                                 deep=self._config.deep_ae,
                                 active_type=self._config.activation_ae,
                                 up_mode=self._config.up_sampling,
                                 down_pool=self._config.down_pooling_ae,
                                 features_sigmoid_active=self._config.features_sigmoid_active)
        else:
            self.ae_net = None

        if self._config.predict_out == 'features':
            predict_out_ch = self.ae_net.n_features_ch
            predict_out_size = self.ae_net.n_features_size
        elif self._config.predict_out == 'images':
            predict_out_ch = 1
            predict_out_size = self._config.image_size
        else:
            raise NameError(f'Nonna valid predict_out type: {self._config.predict_out}')

        # if self._config.predict_type == 'phase':
        #     inter_ch = predict_out_ch
        # else:
        #     inter_ch = 2 * predict_out_ch

        inter_ch = 2 * predict_out_ch
        self.phase_predictor = PhaseRetrievalPredictor(out_ch=predict_out_ch,
                                                       inter_ch=inter_ch,
                                                       out_img_size=predict_out_size,
                                                       fc_multy_coeff=self._config.predict_fc_multy_coeff,
                                                       fft_norm=self._config.fft_norm,
                                                       predict_type=self._config.predict_type,
                                                       im_img_size=self._config.image_size,
                                                       conv_type=self._config.predict_conv_type,
                                                       active_type=self._config.activation_enc,
                                                       deep_fc=5,
                                                       features_sigmoid_active=self._config.features_sigmoid_active,
                                                       use_rfft=self._config.use_rfft)

        if self._config.use_ref_net:
            if self._config.predict_out == 'features':
                in_ch_features = self.ae_net.n_features_ch
            else:
                in_ch_features = None
            self.ref_unet = UNetConv(n_encoder_ch=self.n_encoder_ch,
                                     deep=self._config.deep_ae,
                                     in_ch_features=in_ch_features,
                                     skip_input=self._config.ref_net_skip_input,
                                     active_type=self._config.activation_refnet,
                                     up_mode=self._config.up_sampling,
                                     down_pool=self._config.down_pooling_refnet,
                                     features_sigmoid_active=self._config.features_sigmoid_active)
        else:
            self.ref_unet = None

    def load_module(self, state_dict: Dict[str, Tensor], module: nn.Module, name: str, force: bool = False) -> bool:
        def is_load_module():
            return (name in state_dict) and \
                   ((self._config.load_modules[0] == 'all') or (name in self._config.load_modules))

        if is_load_module():
            self._log.info(f'Load weights of {name}')
            module.load_state_dict(state_dict[name])
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
        if self.load_module(state_dict, self.ae_net, ModulesNames.ae_model, force) and (self.ae_net is not None):
            loaded_models.append(ModulesNames.ae_model)

        if self.load_module(state_dict, self.ref_unet, ModulesNames.ref_net, force):
            loaded_models.append(ModulesNames.ref_net)

        if self.load_module(state_dict, self.phase_predictor, ModulesNames.magnitude_encoder, force):
            loaded_models.append(ModulesNames.magnitude_encoder)

        return loaded_models

    def get_state_dict(self) -> Dict[str, Any]:
        save_state = {ModulesNames.config: self._config.as_dict(),
                      ModulesNames.magnitude_encoder: self.phase_predictor.state_dict()}
        if self._config.use_ref_net:
            save_state[ModulesNames.ref_net] = self.ref_unet.state_dict()

        if self.ae_net is not None:
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

    def forward_magnitude_encoder(self, data_batch: DataBatch, eval_mode: bool = False) -> InferredBatch:
        if eval_mode:
            self.set_eval_mode()
        features_batch_recon, intermediate_features = self.phase_predictor(data_batch.fft_magnitude)
        if self._config.predict_out == 'features':
            recon_batch = self.ae_net.decode(features_batch_recon)
        elif self._config.predict_out == 'images':
            recon_batch = features_batch_recon

        if self._config.predict_out == 'features':
            feature_encoder = self.ae_net.encode(data_batch.image)
            decoded_batch = self.ae_net.decode(feature_encoder)
        else:
            feature_encoder = None
            decoded_batch = None

        inferred_batch = InferredBatch(img_recon=recon_batch,
                                       feature_recon=features_batch_recon,
                                       feature_encoder=feature_encoder,
                                       decoded_img=decoded_batch,
                                       intermediate_features=intermediate_features)
        if self._config.use_ref_net:
            inferred_batch.img_recon_ref = self.ref_unet(recon_batch.detach(), features_batch_recon.detach())
            inferred_batch.fft_magnitude_recon_ref = self.forward_magnitude_fft(inferred_batch.img_recon_ref)

        if eval_mode:
            self.set_train_mode()
        return inferred_batch

    def forward_ae(self, data_batch: DataBatch, eval_mode: bool = False) -> InferredBatch:
        if eval_mode:
            self.set_eval_mode()
        recon_batch, features_batch = self.ae_net(data_batch.image)
        feature_recon = self.ae_net.encode(recon_batch)
        if eval_mode:
            self.set_train_mode()
        return InferredBatch(img_recon=recon_batch, feature_encoder=features_batch, feature_recon=feature_recon)

    def forward_magnitude_fft(self, data_batch: Tensor) -> Tensor:
        if self._config.use_rfft:
            fft_data_batch = torch.fft.rfft2(data_batch, norm=self._config.fft_norm)
        else:
            fft_data_batch = torch.fft.fft2(data_batch, norm=self._config.fft_norm)
        magnitude_batch = torch.abs(fft_data_batch)
        return magnitude_batch

    def set_eval_mode(self):
        self.phase_predictor.eval()
        if self.ae_net is not None:
            self.ae_net.eval()
        if self._config.use_ref_net:
            self.ref_unet.eval()

    def set_train_mode(self):
        self.phase_predictor.train()
        if self.ae_net is not None:
            self.ae_net.train()
        if self._config.use_ref_net:
            self.ref_unet.train()


