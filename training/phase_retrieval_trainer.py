import torch
import numpy as np
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import fire
import os
import copy
from typing import Optional, List
from tqdm import tqdm
from models import Discriminator
from torch.optim.lr_scheduler import MultiStepLR

from common import LossesPRFeatures, InferredBatch, ConfigTrainer, l2_grad_norm,  LossesGradNorms,  DiscriminatorBatch
from common import im_concatenate, l2_perceptual_loss, PATHS, DataBatch, S3FileSystem

from training.base_phase_retrieval_trainer import BaseTrainerPhaseRetrieval
from training.phase_retrieval_model import PhaseRetrievalAeModel
from training.utils import ModulesNames


class TrainerPhaseRetrievalAeFeatures(BaseTrainerPhaseRetrieval):
    def __init__(self, config: ConfigTrainer, experiment_name: str):
        super(TrainerPhaseRetrievalAeFeatures, self).__init__(config=config, experiment_name=experiment_name)

        self.adv_loss = nn.MSELoss()
        self.l2_loss = nn.MSELoss()
        self.n_epochs_ae = config.n_epochs_ae

        self._generator_model = PhaseRetrievalAeModel(config=config, s3=self._s3, log=self._log)

        if self.config.use_gan:
            if self.config.predict_out == 'features':
                self.features_discriminator = Discriminator(input_ch=self._generator_model.ae_net.n_features_ch,
                                                            in_conv_ch=self._generator_model.ae_net.n_features_ch,
                                                            input_norm_type=self.config.disrim_input_norm,
                                                            fc_norm_type=self.config.disrim_fc_norm,
                                                            img_size=self._generator_model.ae_net.n_features_size,
                                                            n_fc_layers=self.config.disrim_features_fc_layers,
                                                            deep_conv_net=1,
                                                            reduce_validity=True,
                                                            use_res_blocks=False,
                                                            active_type=self.config.activation_discrim)
            else:
                self.features_discriminator = None

            self.img_discriminator = Discriminator(in_conv_ch=self.config.disrim_in_conv_ch,
                                                   input_norm_type=self.config.disrim_input_norm,
                                                   fc_norm_type=self.config.disrim_fc_norm,
                                                   img_size=self.img_size,
                                                   n_fc_layers=self.config.disrim_fc_layers,
                                                   deep_conv_net=3,
                                                   reduce_validity=True,
                                                   active_type=self.config.activation_discrim)
        else:
            self.img_discriminator = None
            self.features_discriminator = None

        self._generator_model.set_train_mode()
        self._generator_model.set_device(self.device)

        if self.config.is_train_ae:
            self.optimizer_ae = optim.Adam(params=self._generator_model.ae_net.parameters(), lr=self.learning_rate)
        else:
            self.optimizer_ae = None

        self.optimizer_en = optim.Adam(params=self._generator_model.get_params(), lr=self.learning_rate)

        if self.config.use_gan:
            if self.config.predict_out == 'features':
                self.features_discriminator.train()
                self.features_discriminator.to(device=self.device)

            self.img_discriminator.train()
            self.img_discriminator.to(device=self.device)
            disrim_params = list(self.img_discriminator.parameters())
            if self.config.predict_out == 'features':
                disrim_params += list(self.features_discriminator.parameters())
            self.optimizer_discr = optim.Adam(params=disrim_params, lr=self.learning_rate)

        else:
            self.optimizer_discr = None

        if self.config.use_gan:
            self._lr_schedulers_en = [
                MultiStepLR(self.optimizer_en, self.config.lr_milestones_en, self.config.lr_reduce_rate_en),
                MultiStepLR(self.optimizer_discr, self.config.lr_milestones_en, self.config.lr_reduce_rate_en)
            ]
        else:
            self._lr_schedulers_en = [
                MultiStepLR(self.optimizer_en, self.config.lr_milestones_en, self.config.lr_reduce_rate_en)
            ]

        if self.config.is_train_ae:
            self._lr_scheduler_ae = MultiStepLR(self.optimizer_ae,
                                                self.config.lr_milestones_ae,
                                                self.config.lr_reduce_rate_ae)
        else:
            self._lr_scheduler_ae = None

        self.load_models()

    def load_models(self):
        def is_load_module(name: str, sate_dist):
            return (name in sate_dist) and \
                   ((self.config.load_modules[0] == 'all') or (name in self.config.load_modules))

        if self.config.path_pretrained is not None:
            if self._s3.is_s3_url(self.config.path_pretrained):
                loaded_sate = self._s3.load_object(self.config.path_pretrained, torch.load)
            else:
                assert os.path.isfile(self.config.path_pretrained)
                loaded_sate = torch.load(self.config.path_pretrained)

            self._generator_model.load_modules(loaded_sate)

            if is_load_module(ModulesNames.img_discriminator, loaded_sate) and self.config.use_gan:
                self._log.info(f'Load weights of {ModulesNames.img_discriminator}')
                self.img_discriminator.load_state_dict(loaded_sate[self.ModulesNames.img_discriminator])

            if is_load_module(self.ModulesNames.features_discriminator, loaded_sate) and \
                    self.config.use_gan and self.config.predict_out == 'features':
                self._log.info(f'Load weights of {ModulesNames.features_discriminator}')
                self.features_discriminator.load_state_dict(loaded_sate[self.ModulesNames.features_discriminator])

            if is_load_module(self.ModulesNames.opt_magnitude_encoder, loaded_sate):
                self._log.info(f'Load weights of {ModulesNames.opt_magnitude_encoder}')
                self.optimizer_en.load_state_dict(loaded_sate[ModulesNames.opt_magnitude_encoder])

            if is_load_module(self.ModulesNames.opt_discriminators, loaded_sate) and self.config.use_gan:
                self._log.info(f'Load weights of {ModulesNames.opt_maopt_discriminatorsgnitude_encoder}')
                self.optimizer_en.load_state_dict(loaded_sate[ModulesNames.opt_discriminators])

            if is_load_module(self.ModulesNames.opt_ae, loaded_sate):
                self._log.info(f'Load weights of {ModulesNames.opt_ae}')
                self.optimizer_ae.load_state_dict(loaded_sate[ModulesNames.opt_ae])

    def train(self) -> (LossesPRFeatures, LossesPRFeatures, LossesPRFeatures):
        train_en_losses, test_en_losses, test_ae_losses = [], [], []
        if self.config.debug_mode:
            losses_dbg_batch_tr, losses_dbg_batch_ts = self._log_en_magnitude_dbg_batch(self.config.use_gan,
                                                                                        self._global_step)

        if self.config.is_train_ae:
            train_ae_losses, test_ae_losses = self.train_ae()

        if self.config.is_train_encoder:
            test_ae_losses = self.test_eval_ae()
            self._log.info(f' AE training: ts err: {test_ae_losses.mean()}')
            train_en_losses, test_en_losses = self.train_en_magnitude()

        return train_en_losses, test_en_losses, test_ae_losses

    def train_en_magnitude(self) -> (LossesPRFeatures, LossesPRFeatures):
        self._global_step = 0
        use_adv_loss = self.config.use_gan  # and epoch > 5
        train_en_losses, test_en_losses = [], []
        self._log.info(f'train_en_magnitude is Staring')
        init_ts_losses = self.test_eval_en_magnitude(use_adv_loss)
        self._add_losses_tensorboard('en-magnitude/test', init_ts_losses, self._global_step)
        for epoch in range(1, self.n_epochs + 1):
            self._log.info(f'Train Epoch{epoch}')
            tr_losses = self.train_epoch_en_magnitude(epoch, use_adv_loss)
            ts_losses = self.test_eval_en_magnitude(use_adv_loss)
            ts_losses.lr = torch.tensor(self._lr_schedulers_en[0].get_last_lr()[0])
            self._add_losses_tensorboard('en-magnitude/test', ts_losses, self._global_step)

            for lr_scheduler in self._lr_schedulers_en:
                lr_scheduler.step()

            train_en_losses.append(tr_losses)
            test_en_losses.append(ts_losses)

            self._log.info(f'Magnitude Encoder Epoch={epoch}, '
                            f'Train Losses: {tr_losses}, '
                            f'Test Losses: {ts_losses} ')

            self._save_gan_models(epoch)
            losses_dbg_batch_tr, losses_dbg_batch_ts = self._log_en_magnitude_dbg_batch(use_adv_loss, self._global_step)
            self._add_losses_tensorboard('dbg-batch-en-magnitude/train', losses_dbg_batch_tr, self._global_step)
            self._add_losses_tensorboard('dbg-batch-en-magnitude/test', losses_dbg_batch_ts, self._global_step)

        train_en_losses = LossesPRFeatures.merge(train_en_losses)
        test_en_losses = LossesPRFeatures.merge(test_en_losses)

        return train_en_losses, test_en_losses

    def train_epoch_ae(self, epoch: int = 0) -> LossesPRFeatures:
        train_losses = []
        p_bar_train_data_loader = tqdm(self.train_paired_loader)
        for batch_idx, data_batch in enumerate(p_bar_train_data_loader):
            data_batch = self.prepare_data_batch(data_batch)

            self.optimizer_ae.zero_grad()

            inferred_batch = self._generator_model.forward_ae(data_batch)
            ae_losses = self._ae_losses(data_batch, inferred_batch)

            ae_losses.total.backward()
            self.optimizer_ae.step()
            train_losses.append(ae_losses.detach())

            if batch_idx % self.log_interval == 0:
                self._log.info(f'Train Epoch: {epoch} '
                               f'[{batch_idx * len(data_batch)}/{len(self.train_paired_loader)} '
                                f'({100. * batch_idx / len(self.train_paired_loader):.0f}%)], ae_losses: {ae_losses}')
                self._add_losses_tensorboard('ae/train', ae_losses, self._global_step)

            if batch_idx % self.config.log_image_interval == 0:
                with torch.no_grad():
                    self._log_ae_train_dbg_batch(self._global_step)

            self._global_step += 1

        train_losses = LossesPRFeatures.merge(train_losses)
        return train_losses

    def train_epoch_en_magnitude(self, epoch: int = 0, use_adv_loss: bool = False) -> LossesPRFeatures:
        train_losses = []
        p_bar_train_data_loader = tqdm(self.train_paired_loader)
        for batch_idx, data_batch in enumerate(p_bar_train_data_loader):
            data_batch = self.prepare_data_batch(data_batch)

            inferred_batch, tr_losses = self._train_step_generator(data_batch, use_adv_loss=use_adv_loss)

            if self.config.use_gan:
                self._train_step_discriminator(data_batch, inferred_batch, tr_losses)

            train_losses.append(tr_losses.detach())

            if batch_idx % self.log_interval == 0:
                l2_grad_magnitude_encoder_norm, _ = l2_grad_norm(self._generator_model.phase_predictor)
                grad_losses = LossesGradNorms(l2_grad_magnitude_encoder=l2_grad_magnitude_encoder_norm)
                if self.config.use_gan:
                    grad_losses.l2_grad_img_discriminator, _ = l2_grad_norm(self.img_discriminator)
                    if self.config.predict_out == 'features':
                        grad_losses.l2_grad_features_discriminator, _ = l2_grad_norm(self.features_discriminator)

                self._log.info(f'Train Epoch: {epoch} [{batch_idx * len(data_batch)}/{len(self.train_paired_loader.dataset)}'
                                f'({100. * batch_idx / len(self.train_paired_loader):.0f}%)], {train_losses[batch_idx]}, '
                                f'{grad_losses}')
                self._add_losses_tensorboard('en-magnitude/train', tr_losses, self._global_step)
                self._add_losses_tensorboard('en-magnitude/train', grad_losses, self._global_step)

            if batch_idx % self.config.log_image_interval == 0:
                losses_dbg_batch_tr, losses_dbg_batch_ts = self._log_en_magnitude_dbg_batch(use_adv_loss,
                                                                                            self._global_step)
                self._add_losses_tensorboard('dbg-batch-en-magnitude/train', losses_dbg_batch_tr, self._global_step)
                self._add_losses_tensorboard('dbg-batch-en-magnitude/test', losses_dbg_batch_ts, self._global_step)

            self._global_step += 1

        train_losses = LossesPRFeatures.merge(train_losses).mean()
        return train_losses

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
            fft_magnitude_recon = self.forward_magnitude_fft(inferred_batch.img_recon)

            l2_magnitude_loss = self.l2_loss(data_batch.fft_magnitude.detach(), fft_magnitude_recon)
            l2_realness_features = 0.5 * torch.mean(torch.square(inferred_batch.intermediate_features.imag.abs()))
            l1_sparsity_features = torch.mean(inferred_batch.feature_recon.abs())
            img_adv_loss = self.adv_loss(self.img_discriminator(inferred_batch.img_recon).validity,
                                         real_labels)

            if self.config.predict_out == 'features':
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

    def _discrim_ls_loss(self, discriminator: nn.Module, real_img: Tensor, generated_img: Tensor,
                         real_labels: Tensor, fake_labels: Tensor) -> Tensor:
        real_loss = self.adv_loss(discriminator(real_img.detach()).validity, real_labels)
        fake_loss = self.adv_loss(discriminator(generated_img.detach()).validity, fake_labels)
        disrm_loss = 0.5 * (real_loss + fake_loss)
        return disrm_loss

    def _train_step_discriminator(self, data_batch: DataBatch, inferred_batch: InferredBatch, tr_losses: LossesPRFeatures):
        self.optimizer_discr.zero_grad()
        # Measure discriminator's ability to classify real from generated samples
        batch_size = data_batch.image.shape[0]
        real_labels = torch.ones((batch_size, 1), device=self.device, dtype=data_batch.image.dtype)
        fake_labels = torch.zeros((batch_size, 1), device=self.device, dtype=data_batch.image.dtype)

        img_disrm_loss = self._discrim_ls_loss(self.img_discriminator,
                                               real_img=inferred_batch.decoded_img,
                                               generated_img=inferred_batch.img_recon,
                                               real_labels=real_labels,
                                               fake_labels=fake_labels)
        tr_losses.disrm_loss = self.config.lambda_discrim_img * img_disrm_loss

        if self.config.use_ref_net:
            ref_img_disrm_loss = self._discrim_ls_loss(self.img_discriminator,
                                                       real_img=inferred_batch.decoded_img,
                                                       generated_img=inferred_batch.img_recon_ref,
                                                       real_labels=real_labels,
                                                       fake_labels=fake_labels)
            tr_losses.disrm_loss += self.config.lambda_discrim_ref_img * ref_img_disrm_loss
        else:
            ref_img_disrm_loss = None

        if self.config.predict_out == 'features':
            feature_disrm_loss = self._discrim_ls_loss(self.features_discriminator,
                                                       real_img=inferred_batch.feature_encoder,
                                                       generated_img=inferred_batch.feature_recon,
                                                       real_labels=real_labels,
                                                       fake_labels=fake_labels)
            tr_losses.disrm_loss += self.config.lambda_discrim_features * feature_disrm_loss
        else:
            feature_disrm_loss = None

        tr_losses.img_disrm_loss = img_disrm_loss
        tr_losses.ref_img_disrm_loss = ref_img_disrm_loss
        tr_losses.features_disrm_loss = feature_disrm_loss
        tr_losses.disrm_loss.backward()
        if self.config.clip_discriminator_grad:
            torch.nn.utils.clip_grad_norm_(self.img_discriminator.parameters(),
                                           self.config.clip_discriminator_grad)
        self.optimizer_discr.step()

    def _train_step_generator(self, data_batch: DataBatch,
                              use_adv_loss: bool = False) -> (InferredBatch, LossesPRFeatures):
        self.optimizer_en.zero_grad()
        inferred_batch = self._generator_model.forward_magnitude_encoder(data_batch)
        tr_losses = self._encoder_losses(data_batch, inferred_batch, use_adv_loss=use_adv_loss)
        tr_losses.total.backward()
        if self.config.clip_encoder_grad is not None:
            torch.nn.utils.clip_grad_norm_(self._generator_model.phase_predictor.parameters(),
                                           self.config.clip_encoder_grad)
        self.optimizer_en.step()

        return inferred_batch, tr_losses

    def _ae_losses(self, data_batch: DataBatch, inferred_batch: InferredBatch) -> LossesPRFeatures:
        fft_magnitude_recon = self.forward_magnitude_fft(inferred_batch.img_recon)
        l2_img_loss = self.l2_loss(data_batch.image, inferred_batch.img_recon)
        l1_sparsity_features = torch.mean(inferred_batch.feature_recon.abs())
        l2_magnitude_loss = self.l2_loss(data_batch.fft_magnitude.detach(), fft_magnitude_recon)
        l2_features_loss = self.l2_loss(inferred_batch.feature_encoder, inferred_batch.feature_recon)
        total_loss = l2_img_loss + l2_features_loss + 0.5 * self.config.lambda_sparsity_features * l1_sparsity_features
        losses = LossesPRFeatures(total=total_loss,
                                  l2_img=l2_img_loss,
                                  l2_features=l2_features_loss,
                                  l2_magnitude=l2_magnitude_loss,
                                  l1_sparsity_features=l1_sparsity_features)

        self._recon_statistics_metrics(inferred_batch, losses)

        return losses

    def _encoder_losses(self, data_batch: DataBatch,
                        inferred_batch: InferredBatch,
                        use_adv_loss: bool = False) -> LossesPRFeatures:

        is_paired = data_batch.is_paired

        fft_magnitude_recon = self.forward_magnitude_fft(inferred_batch.img_recon)
        total_loss = torch.zeros(1, device=self.device)[0]
        l2_img_recon_loss = self.l2_loss(data_batch.image, inferred_batch.img_recon)
        if self.config.predict_out == 'features':
            l2_features_loss = self.l2_loss(inferred_batch.feature_encoder, inferred_batch.feature_recon)
            l1_sparsity_features = torch.mean(inferred_batch.feature_recon.abs())
        else:
            l2_features_loss = None
            l1_sparsity_features = None

        l2_magnitude_loss = self.l2_loss(data_batch.fft_magnitude.detach(), fft_magnitude_recon)
        l2_features_realness = 0.5 * torch.mean(torch.square(inferred_batch.intermediate_features.imag.abs()))

        if self.config.use_ref_net:
            l2_ref_img_recon_loss = self.l2_loss(data_batch.image, inferred_batch.img_recon_ref)
            l2_ref_magnitude_loss = self.l2_loss(data_batch.fft_magnitude.detach(),
                                                 inferred_batch.fft_magnitude_recon_ref)
        else:
            l2_ref_img_recon_loss = None
            l2_ref_magnitude_loss = None

        real_labels = torch.ones((data_batch.fft_magnitude.shape[0], 1),
                                 device=self.device,
                                 dtype=data_batch.fft_magnitude.dtype)

        total_loss += self.config.lambda_img_recon_loss * l2_img_recon_loss
        if use_adv_loss:
            gen_img_discrim_batch: DiscriminatorBatch = self.img_discriminator(inferred_batch.img_recon)
            img_adv_loss = self.adv_loss(gen_img_discrim_batch.validity, real_labels)
            total_loss += self.config.lambda_img_adv_loss * img_adv_loss

            real_img_discrim_batch: DiscriminatorBatch = self.img_discriminator(data_batch.image)
            p_loss_discrim_img = l2_perceptual_loss(gen_img_discrim_batch.features, real_img_discrim_batch.features,
                                                    weights=self.config.weights_plos)
            total_loss += self.config.lambda_img_perceptual_loss * p_loss_discrim_img

            if self.config.use_ref_net:
                gen_ref_img_discrim_batch: DiscriminatorBatch = self.img_discriminator(inferred_batch.img_recon_ref)
                ref_img_adv_loss = self.adv_loss(gen_ref_img_discrim_batch.validity, real_labels)
                total_loss += self.config.lambda_ref_img_adv_loss * ref_img_adv_loss

                p_loss_discrim_ref_img = l2_perceptual_loss(gen_ref_img_discrim_batch.features,
                                                            real_img_discrim_batch.features,
                                                        weights=self.config.weights_plos)
                total_loss += self.config.lambda_ref_img_perceptual_loss * p_loss_discrim_ref_img
            else:
                ref_img_adv_loss = None
                p_loss_discrim_ref_img = None

            if self.config.predict_out == 'features':
                f_disc_generated_batch: DiscriminatorBatch = self.features_discriminator(inferred_batch.feature_recon)
                f_disc_real_batch: DiscriminatorBatch = self.features_discriminator(inferred_batch.feature_encoder)
                p_loss_discrim_f = l2_perceptual_loss(f_disc_generated_batch.features, f_disc_real_batch.features,
                                                      weights=self.config.weights_plos)
                features_adv_loss = self.adv_loss(f_disc_generated_batch.validity, real_labels)

                total_loss += self.config.lambda_features_adv_loss * features_adv_loss + \
                              self.config.lambda_features_perceptual_loss * p_loss_discrim_f
            else:
                p_loss_discrim_f = None
                features_adv_loss = None
        else:
            img_adv_loss = None
            features_adv_loss = None
            p_loss_discrim_f = None
            p_loss_discrim_img = None
            p_loss_discrim_ref_img = None

        total_loss += self.config.lambda_magnitude_recon_loss * l2_magnitude_loss + \
                      self.config.lambda_features_realness * l2_features_realness

        if self.config.predict_out == 'features':
            total_loss += self.config.lambda_features_recon_loss * l2_features_loss + \
                          self.config.lambda_sparsity_features * l1_sparsity_features

        if self.config.use_ref_net:
            total_loss += self.config.lambda_ref_magnitude_recon_loss * l2_ref_magnitude_loss + \
                          self.config.lambda_img_recon_loss * l2_ref_img_recon_loss

        losses = LossesPRFeatures(total=total_loss,
                                  l2_img=l2_img_recon_loss,
                                  l2_ref_img=l2_ref_img_recon_loss,
                                  l2_features=l2_features_loss,
                                  l2_magnitude=l2_magnitude_loss,
                                  l2_ref_magnitude=l2_ref_magnitude_loss,
                                  l1_sparsity_features=l1_sparsity_features,
                                  realness_features=l2_features_realness,
                                  img_adv_loss=img_adv_loss,
                                  ref_img_adv_loss=ref_img_adv_loss,
                                  features_adv_loss=features_adv_loss,
                                  perceptual_disrim_features=p_loss_discrim_f,
                                  perceptual_disrim_img=p_loss_discrim_img,
                                  perceptual_disrim_ref_img=p_loss_discrim_ref_img)

        self._recon_statistics_metrics(inferred_batch, losses)

        if self.config.use_ref_net:
            losses.mean_img_ref = inferred_batch.img_recon_ref.mean()
            losses.std_img_ref = inferred_batch.img_recon_ref.std()
            losses.min_img_ref = inferred_batch.img_recon_ref.min()
            losses.max_img_ref = inferred_batch.img_recon_ref.max()

        return losses

    @staticmethod
    def _recon_statistics_metrics(inferred_batch: InferredBatch, losses: LossesPRFeatures):
        losses.mean_img = inferred_batch.img_recon.mean()
        losses.std_img = inferred_batch.img_recon.std()
        losses.min_img = inferred_batch.img_recon.min()
        losses.max_img = inferred_batch.img_recon.max()
        losses.mean_features = inferred_batch.feature_recon.mean()
        losses.std_features = inferred_batch.feature_recon.std()
        losses.min_features = inferred_batch.feature_recon.min()
        losses.max_features = inferred_batch.feature_recon.max()

    def test_eval_ae(self) -> LossesPRFeatures:
        ae_losses = []
        self._generator_model.set_eval_mode()
        with torch.no_grad():
            for batch_idx, data_batch in enumerate(self.test_loader):
                data_batch = self.prepare_data_batch(data_batch)
                inferred_batch = self._generator_model.forward_ae(data_batch)
                ae_losses_batch = self._ae_losses(data_batch, inferred_batch)
                ae_losses.append(ae_losses_batch)
            ae_losses = LossesPRFeatures.merge(ae_losses)
        self._generator_model.set_train_mode()
        return ae_losses

    def test_eval_en_magnitude(self, use_adv_loss: bool = False) -> LossesPRFeatures:
        losses_ts = []
        with torch.no_grad():
            for batch_idx, data_batch in enumerate(self.test_loader):
                data_batch = self.prepare_data_batch(data_batch)
                inferred_batch = self._generator_model.forward_magnitude_encoder(data_batch, eval_mode=False)
                losses_ts_ = self._encoder_losses(data_batch, inferred_batch, use_adv_loss=use_adv_loss)
                losses_ts.append(losses_ts_.detach())

            losses_ts = LossesPRFeatures.merge(losses_ts).mean()
            return losses_ts

    def train_ae(self) -> (LossesPRFeatures, LossesPRFeatures):
        self._global_step = 0
        tr_losses = []
        ts_losses = []
        init_ae_losses = self.test_eval_ae()
        init_ae_losses = init_ae_losses.mean()
        self._log.info(f' AE training: init ts losses: {init_ae_losses}')
        self._log_ae_train_dbg_batch(0)
        self._add_losses_tensorboard('ae/test', init_ae_losses, self._global_step)
        for epoch in range(1, self.n_epochs_ae + 1):
            tr_losses_epoch = self.train_epoch_ae(epoch)
            tr_losses_epoch = tr_losses_epoch.mean()

            ts_losses_epoch = self.test_eval_ae()
            ts_losses_epoch.lr = torch.tensor(self._lr_scheduler_ae.get_last_lr()[0])

            self._lr_scheduler_ae.step()
            self._add_losses_tensorboard('ae/test', ts_losses_epoch, self._global_step)
            self._log.info(f'AE training: Epoch {epoch}, '
                           f'l2_recon_err_tr: {tr_losses_epoch}, '
                            f'l2_recon_err_ts: {ts_losses_epoch}')
            with torch.no_grad():
                self._log_ae_train_dbg_batch(self._global_step)

                if self.models_path is not None:
                    ae_state = {'ae_model': self._generator_model.ae_net.state_dict(),
                                'opt_ae': self.optimizer_ae.state_dict()}
                    torch.save(ae_state, os.path.join(self.models_path, f'ae_model.pt'))
            tr_losses.append(tr_losses_epoch)
            ts_losses.append(ts_losses_epoch)
        tr_losses = LossesPRFeatures.merge(tr_losses)
        ts_losses = LossesPRFeatures.merge(ts_losses)
        return tr_losses, ts_losses

    def _save_gan_models(self, step: int) -> None:
        if self.models_path is not None:
            save_state = self._generator_model.get_state_dict()
            save_state['opt_magnitude_encoder'] = self.optimizer_en.state_dict()
            if self.config.use_gan:
                save_state['img_discriminator'] = self.img_discriminator.state_dict()
                save_state['opt_discr'] = self.optimizer_discr.state_dict()
                if self.config.predict_out == 'features':
                    save_state['features_discriminator'] = self.features_discriminator.state_dict()
            if self.config.is_train_ae:
                save_state['opt_ae'] = self.optimizer_ae.state_dict()
            torch.save(save_state, os.path.join(self.models_path, f'phase-retrieval-gan-model.pt'))

    def _log_ae_train_dbg_batch(self, step: Optional[int] = None) -> None:
        if not step:
            step = self._global_step

        with torch.no_grad():
            recon_data_tr_batch = self._generator_model.forward_ae(self.data_tr_batch, eval_mode=False)
            recon_data_ts_batch = self._generator_model.forward_ae(self.data_ts_batch, eval_mode=False)

        img_grid_tr = self._grid_images(self.data_tr_batch, recon_data_tr_batch)
        img_grid_ts = self._grid_images(self.data_ts_batch, recon_data_ts_batch)
        features_grid_tr = self._grid_features(recon_data_tr_batch)
        features_grid_ts = self._grid_features(recon_data_ts_batch)

        self.log_image_grid(img_grid_tr, 'train-ae/recon', step)
        self.log_image_grid(img_grid_ts, 'test-ae/recon', step)
        self.log_image_grid(features_grid_tr, 'train-ae/features', step)
        self.log_image_grid(features_grid_ts, 'test-ae/features', step)

    def _log_en_magnitude_dbg_batch(self, use_adv_loss: bool, step: int = None) -> (LossesPRFeatures, LossesPRFeatures):
        if not step:
            step = self._global_step
        with torch.no_grad():
            inferred_batch_tr = self._generator_model.forward_magnitude_encoder(self.data_tr_batch, eval_mode=False)
            inferred_batch_ts = self._generator_model.forward_magnitude_encoder(self.data_ts_batch, eval_mode=False)

            tr_losses = self._encoder_losses(self.data_tr_batch, inferred_batch_tr, use_adv_loss=use_adv_loss)
            ts_losses = self._encoder_losses(self.data_ts_batch, inferred_batch_ts, use_adv_loss=use_adv_loss)

            # tr_mse_img = np.mean((self.prepare_dbg_batch(self.data_tr_batch.image - inferred_batch_tr.img_recon))**2)
            # tr_mse_img_ref = np.mean((self.prepare_dbg_batch(self.data_tr_batch.image -
            #                                                  inferred_batch_tr.img_recon_ref)) ** 2)
            # tr_losses.l2_img_np = tr_mse_img
            # tr_losses.l2_ref_img_np = tr_mse_img_ref

            # ts_mse_img = np.mean((self.prepare_dbg_batch(self.data_ts_batch.image - inferred_batch_ts.img_recon))**2)
            # ts_mse_img_ref = np.mean((self.prepare_dbg_batch(self.data_ts_batch.image -
            #                                                  inferred_batch_ts.img_recon_ref)) ** 2)
            # ts_losses.l2_img_np = ts_mse_img
            # ts_losses.l2_ref_img_np = ts_mse_img_ref

            img_grid_grid_tr = self._grid_images(self.data_tr_batch, inferred_batch_tr)
            img_grid_grid_ts = self._grid_images(self.data_ts_batch, inferred_batch_ts)

            features_grid_grid_tr = self._grid_features(inferred_batch_tr)
            features_grid_grid_ts = self._grid_features(inferred_batch_ts)

            # recon_data_tr_batch, recon_data_ts_batch = self.get_dbg_data(inferred_batch_tr, inferred_batch_ts)
            #
            # features_recon_tr = self.prepare_dbg_batch(inferred_batch_tr.feature_recon[:self.config.dbg_features_batch],
            #                                            grid_ch=True)
            # features_recon_ts = self.prepare_dbg_batch(inferred_batch_ts.feature_recon[:self.config.dbg_features_batch],
            #                                            grid_ch=True)
            #
            # features_enc_tr = self.prepare_dbg_batch(inferred_batch_tr.feature_encoder[:self.config.dbg_features_batch],
            #                                          grid_ch=True)
            # features_enc_ts = self.prepare_dbg_batch(inferred_batch_ts.feature_encoder[:self.config.dbg_features_batch],
            #                                          grid_ch=True)

            self.log_image_grid(img_grid_grid_tr, tag_name='train_en_magnitude/img-origin-autoencoded-recon', step=step)
            self.log_image_grid(img_grid_grid_ts, tag_name='test_en_magnitude/img-origin-autoencoded-recon', step=step)

            self.log_image_grid(features_grid_grid_tr, tag_name='train_en_magnitude/features-origin-recon', step=step)
            self.log_image_grid(features_grid_grid_ts, tag_name='test_en_magnitude/features-origin-recon', step=step)

            # self._log_images(recon_data_tr_batch, tag_name='train_en_magnitude/recon', step=step)
            # self._log_images(recon_data_ts_batch, tag_name='test_en_magnitude/recon', step=step)

            # self._log_images(features_recon_tr, tag_name='train_en_magnitude/features_recon', step=step)
            # self._log_images(features_recon_ts, tag_name='test_en_magnitude/features_recon', step=step)
            #
            # self._log_images(features_enc_tr, tag_name='train_en_magnitude/features_encoder', step=step)
            # self._log_images(features_enc_ts, tag_name='test_en_magnitude/features_encoder', step=step)
        return tr_losses, ts_losses

    def get_dbg_batch(self, data_batch: DataBatch, inferred_batch: InferredBatch) -> np.ndarray:
        data_batch = self.prepare_dbg_batch(data_batch.image)
        recon_batch = self.prepare_dbg_batch(inferred_batch.img_recon)
        dbg_data_batch = [data_batch]
        if inferred_batch.decoded_img is not None:
            dbg_data_batch.append(self.prepare_dbg_batch(inferred_batch.decoded_img))
        dbg_data_batch.append(recon_batch)

        if inferred_batch.img_recon_ref is not None:
            ref_recon_batch = self.prepare_dbg_batch(inferred_batch.img_recon_ref)
            dbg_data_batch.append(ref_recon_batch)

        dbg_data_batch = im_concatenate(dbg_data_batch, axis=0)
        return dbg_data_batch


def run_ae_features_trainer(experiment_name: str = 'recon-l2-ae',
                            config_path: str = None,
                            train_model: bool = True,
                            fit_batch: bool = False,
                            **kwargs):


    if config_path is None:
        config = ConfigTrainer()
    elif S3FileSystem.is_s3_url(config_path):
        config = S3FileSystem().load_object(config_path, ConfigTrainer.from_data_file)
    else:
        assert os.path.exists(config_path)
        config = ConfigTrainer.from_data_file(config_path)

    config.log_path = PATHS.LOG
    config = config.update(**kwargs)

    trainer = TrainerPhaseRetrievalAeFeatures(config=config, experiment_name=experiment_name)
    task_s3_path = trainer.get_task_s3_path()

    if train_model:
        train_en_losses, test_en_losses, test_ae_losses = trainer.train()

    if fit_batch:
        trainer._log.info(f'FITTING TEST BATCH')

        for num_img_fit in range(len(trainer.data_ts_batch)):
            trainer._log.info(f'fit ts batch num - {num_img_fit}')
            fit_batch = torch.unsqueeze(trainer.data_ts_batch[num_img_fit], 0)
            data_ts_batch_fitted, losses_fit = trainer.fit(fit_batch,
                                                           lr=0.00001, n_iter=1000,
                                                           lr_milestones=[250, 5000, 750],
                                                           lr_reduce_rate=0.5,
                                                           name=str(num_img_fit))
    return task_s3_path


if __name__ == '__main__':
    fire.Fire(run_ae_features_trainer)
