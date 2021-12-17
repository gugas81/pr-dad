
import torch
import torch.nn as nn
import torch.optim as optim
import fire
import copy
from typing import Optional, List
from tqdm import tqdm
from models import Discriminator
from torch.optim.lr_scheduler import MultiStepLR

from common import LossesPRFeatures, InferredBatch, ConfigTrainer, l2_grad_norm,  LossesGradNorms,  DiscriminatorBatch
from common import im_concatenate, l2_perceptual_loss, DataBatch, LossImg

from training.base_phase_retrieval_trainer import BaseTrainerPhaseRetrieval
from training.phase_retrieval_trainer import TrainerPhaseRetrievalAeFeatures
from training.phase_retrieval_model import PhaseRetrievalAeModel
from training.utils import ModulesNames


class TrainerPhaseFitter(TrainerPhaseRetrievalAeFeatures):
    def __init__(self, config: ConfigTrainer, experiment_name: str):
        BaseTrainerPhaseRetrieval.__init__(self,config=config, experiment_name=experiment_name)

        self.adv_loss = nn.MSELoss()

        self.l2_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

        self.l2_img_loss = LossImg(loss_type='l2', rot180=config.loss_rot180, device=self.device)
        self.l1_img_loss = LossImg(loss_type='l1', rot180=config.loss_rot180, device=self.device)

        if self._config.use_lpips:
            self._lpips_loss = LossImg(loss_type='lpips', rot180=config.loss_rot180, device=self.device)
        else:
            self._lpips_loss = None

        self._generator_model = PhaseRetrievalAeModel(config=self._config, s3=self._s3, log=self._log)

        self._init_discriminators()

    def fit(self, data_batch: DataBatch, lr: float, lr_milestones: List[int], lr_reduce_rate: float, n_iter: int,
            name: str) -> (InferredBatch, LossesPRFeatures):
        self._global_step = 0
        init_params_state_state = copy.deepcopy(self._generator_model.phase_predictor.state_dict())
        fit_optimizer = optim.Adam(params=self._generator_model.phase_predictor.parameters(), lr=lr)
        lr_scheduler_fitter = MultiStepLR(fit_optimizer, lr_milestones, lr_reduce_rate)

        real_labels = torch.ones((data_batch.image.shape[0], 1), device=self.device, dtype=data_batch.image.dtype)
        losses_fit = []
        for ind_iter in tqdm(range(n_iter)):
            fit_optimizer.zero_grad()
            inferred_batch = self._generator_model.forward_magnitude_encoder(data_batch)
            fft_magnitude_recon = self._generator_model.forward_magnitude_fft(inferred_batch.img_recon)

            l2_magnitude_loss = self.l2_loss(data_batch.fft_magnitude.detach(), fft_magnitude_recon)
            l2_realness_features = 0.5 * torch.mean(torch.square(inferred_batch.intermediate_features.imag.abs()))
            l1_sparsity_features = torch.mean(inferred_batch.feature_recon.abs())
            img_adv_loss = self.adv_loss(self.img_discriminator(inferred_batch.img_recon).validity,
                                         real_labels)

            if self._config.predict_out == 'features':
                features_adv_loss = self.adv_loss(self.features_discriminator(inferred_batch.feature_recon).validity,
                                                  real_labels)
                total_loss = 0.01 * features_adv_loss
            else:
                features_adv_loss = None

            total_loss += 2.0 * l2_magnitude_loss + 1.0 * l2_realness_features + 0.01 * l1_sparsity_features + \
                          0.01 * img_adv_loss

            losses = LossesPRFeatures(total=total_loss,
                                      l2_magnitude=l2_magnitude_loss,
                                      l1_sparsity_features=l1_sparsity_features,
                                      realness_features=l2_realness_features,
                                      img_adv_loss=img_adv_loss,
                                      features_adv_loss=features_adv_loss,
                                      lr=torch.tensor(lr_scheduler_fitter.get_last_lr()[0]))

            losses_fit.append(losses.detach())

            total_loss.backward()
            fit_optimizer.step()
            lr_scheduler_fitter.step()

            if (ind_iter % 50 == 0) or (ind_iter == n_iter - 1):
                l2_grad_encoder_norm, _ = l2_grad_norm(self._generator_model.phase_predictor)
                l2_grad = LossesGradNorms(l2_grad_magnitude_encoder=l2_grad_encoder_norm)
                self._log.info(f'Fitter: iter={ind_iter}, {losses}, {l2_grad}')
                self._add_losses_tensorboard(f'fit-{name}', losses, step=self._global_step)
                self._add_losses_tensorboard(f'fit-{name}', l2_grad, step=self._global_step)

            if (ind_iter % 100 == 0) or (ind_iter == n_iter - 1):
                dbg_data_batch = self.get_dbg_batch(data_batch, inferred_batch)
                dbg_enc_features_batch = self.prepare_dbg_batch(inferred_batch.feature_encoder, grid_ch=True)
                self._log_images(dbg_data_batch, tag_name=f'fit/{name}-magnitude-recon', step=self._global_step)
                self._log_images(dbg_enc_features_batch, tag_name=f'fit/{name}-features_encode', step=self._global_step)

            self._global_step += 1

        losses_fit = LossesPRFeatures.merge(losses_fit)
        self._generator_model.phase_predictor.load_state_dict(init_params_state_state)

        return inferred_batch, losses_fit