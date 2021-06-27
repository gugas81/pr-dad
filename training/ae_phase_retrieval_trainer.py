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
from models import PhaseRetrievalPredictor,  Discriminator, AeConv
from torch.optim.lr_scheduler import MultiStepLR
from dataclasses import dataclass

from common import LossesPRFeatures, InferredBatch, ConfigTrainer, l2_grad_norm,  LossesGradNorms,  DiscriminatorBatch
from common import im_concatenate, S3FileSystem, l2_perceptual_loss, PATHS

from training.base_phase_retrieval_trainer import TrainerPhaseRetrieval


class TrainerPhaseRetrievalAeFeatures(TrainerPhaseRetrieval):
    @dataclass
    class ModulesNames:
        ae_model = 'ae_model'
        magnitude_encoder = 'magnitude_encoder'
        img_discriminator = 'img_discriminator'
        features_discriminator = 'features_discriminator'
        opt_magnitude_encoder = 'opt_magnitude_encoder'
        opt_discriminators = 'opt_discriminators'
        opt_ae = 'opt_ae'

    def __init__(self, config: ConfigTrainer, experiment_name: str):
        super(TrainerPhaseRetrievalAeFeatures, self).__init__(config=config, experiment_name=experiment_name)

        self.adv_loss = nn.MSELoss()
        self.l2_loss = nn.MSELoss()

        self.n_epochs_ae = config.n_epochs_ae
        self.n_encoder_ch = config.n_features // 4
        self.ae_net = AeConv(n_encoder_ch=self.n_encoder_ch, img_size=self.img_size, deep=self.config.deep_ae)

        if self.config.predict_out == 'features':
            predict_out_ch = self.ae_net.n_features_ch
            predict_out_size = self.ae_net.n_features_size
        elif self.config.predict_out == 'images':
            predict_out_ch = 1
            predict_out_size = self.config.image_size
        else:
            raise NameError(f'Nonna valid predict_out type: {self.config.predict_out}')

        self.phase_predictor = PhaseRetrievalPredictor(out_ch=predict_out_ch,
                                                       inter_ch=2*predict_out_ch,
                                                       out_img_size=predict_out_size,
                                                       fc_multy_coeff=self.config.predict_fc_multy_coeff,
                                                       fft_norm=self.config.fft_norm,
                                                       predict_type=self.config.predict_type,
                                                       im_img_size=self.img_size,
                                                       conv_type=self.config.predict_conv_type)

        if self.config.use_gan:
            if self.config.predict_out == 'features':
                self.features_discriminator = Discriminator(input_ch=self.ae_net.n_features_ch,
                                                            in_conv_ch=2*self.ae_net.n_features_ch,
                                                            input_norm_type=self.config.disrim_input_norm,
                                                            fc_norm_type=self.config.disrim_fc_norm,
                                                            img_size=self.ae_net.n_features_size,
                                                            n_fc_layers=self.config.disrim_fc_layers,
                                                            deep_conv_net=1,
                                                            reduce_validity=True)
            else:
                self.features_discriminator = None

            self.img_discriminator = Discriminator(in_conv_ch=self.config.disrim_in_conv_ch,
                                                   input_norm_type=self.config.disrim_input_norm,
                                                   fc_norm_type=self.config.disrim_fc_norm,
                                                   img_size=self.img_size,
                                                   n_fc_layers=self.config.disrim_fc_layers,
                                                   deep_conv_net=3,
                                                   reduce_validity=True)
        else:
            self.img_discriminator = None
            self.features_discriminator = None

        if self.config.is_train_ae:
            self.ae_net.train()
        self.ae_net.to(device=self.device)

        self.phase_predictor.train()
        self.phase_predictor.to(device=self.device)

        if self.config.is_train_ae:
            self.optimizer_ae = optim.Adam(params=self.ae_net.parameters(), lr=self.learning_rate)
        else:
            self.optimizer_ae = None
        self.optimizer_en = optim.Adam(params=self.phase_predictor.parameters(), lr=self.learning_rate)

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

        self._lr_schedulers_en = [
            MultiStepLR(self.optimizer_en, self.config.lr_milestones_en, self.config.lr_reduce_rate_en),
            MultiStepLR(self.optimizer_discr, self.config.lr_milestones_en, self.config.lr_reduce_rate_en)
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
            if S3FileSystem.is_s3_url(self.config.path_pretrained):
                loaded_sate = S3FileSystem().load_object(self.config.path_pretrained, torch.load)
            else:
                assert os.path.isfile(self.config.path_pretrained)
                loaded_sate = torch.load(self.config.path_pretrained)
            if is_load_module(self.ModulesNames.ae_model, loaded_sate):
                self._log.info(f'Load wieghts of {self.ModulesNames.ae_model}')
                self.ae_net.load_state_dict(loaded_sate[self.ModulesNames.ae_model])

            if is_load_module(self.ModulesNames.magnitude_encoder, loaded_sate):
                self._log.info(f'Load wieghts of {self.ModulesNames.magnitude_encoder}')
                self.phase_predictor.load_state_dict(loaded_sate[self.ModulesNames.magnitude_encoder])

            if is_load_module(self.ModulesNames.img_discriminator, loaded_sate) and self.config.use_gan:
                self._log.info(f'Load wieghts of {self.ModulesNames.img_discriminator}')
                self.img_discriminator.load_state_dict(loaded_sate[self.ModulesNames.img_discriminator])

            if is_load_module(self.ModulesNames.features_discriminator, loaded_sate) and \
                    self.config.use_gan and self.config.predict_out == 'features':
                self._log.info(f'Load wieghts of {self.ModulesNames.features_discriminator}')
                self.features_discriminator.load_state_dict(loaded_sate[self.ModulesNames.features_discriminator])

            if is_load_module(self.ModulesNames.opt_magnitude_encoder, loaded_sate):
                self._log.info(f'Load wieghts of {self.ModulesNames.opt_magnitude_encoder}')
                self.optimizer_en.load_state_dict(loaded_sate[self.ModulesNames.opt_magnitude_encoder])

            if is_load_module(self.ModulesNames.opt_discriminators, loaded_sate) and self.config.use_gan:
                self._log.info(f'Load wieghts of {self.ModulesNames.opt_maopt_discriminatorsgnitude_encoder}')
                self.optimizer_en.load_state_dict(loaded_sate[self.ModulesNames.opt_discriminators])

            if is_load_module(self.ModulesNames.opt_ae, loaded_sate):
                self._log.info(f'Load wieghts of {self.ModulesNames.opt_ae}')
                self.optimizer_ae.load_state_dict(loaded_sate[self.ModulesNames.opt_ae])

    def forward_magnitude_encoder(self, data_batch: Tensor, eval_mode: bool = False) -> InferredBatch:
        if eval_mode:
            self.set_eval_mode()
        magnitude_batch = self.forward_magnitude_fft(data_batch)
        features_batch_recon, intermediate_features = self.phase_predictor(magnitude_batch)
        if self.config.predict_out == 'features':
            recon_batch = self.ae_net.decoder(features_batch_recon)
        elif self.config.predict_out == 'images':
            recon_batch = features_batch_recon

        feature_encoder = self.ae_net.encoder(data_batch)
        decoded_batch = self.ae_net.decoder(feature_encoder)
        inferred_batch = InferredBatch(fft_magnitude=magnitude_batch,
                                       img_recon=recon_batch,
                                       feature_recon=features_batch_recon,
                                       feature_encoder=feature_encoder,
                                       decoded_img=decoded_batch,
                                       intermediate_features=intermediate_features)
        if eval_mode:
            self.set_train_model()
        return inferred_batch

    def set_eval_mode(self):
        self.phase_predictor.eval()
        self.ae_net.eval()

    def set_train_model(self):
        self.phase_predictor.train()
        self.ae_net.train()

    def forward_ae(self, data_batch: Tensor, eval_mode: bool = False) -> InferredBatch:
        if eval_mode:
            self.set_eval_mode()
        fft_magnitude = self.forward_magnitude_fft(data_batch)
        recon_batch, features_batch = self.ae_net(data_batch)
        feature_recon = self.ae_net.encoder(recon_batch)
        if eval_mode:
            self.set_train_model()
        return InferredBatch(img_recon=recon_batch, feature_encoder=features_batch,
                             fft_magnitude=fft_magnitude, feature_recon=feature_recon)

    def train(self) -> (LossesPRFeatures, LossesPRFeatures, LossesPRFeatures):
        if self.config.is_train_ae:
            self.train_ae()

        test_ae_losses = self.test_eval_ae()

        self._log.info(f' AE training: ts err: {test_ae_losses}')

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

    def _log_en_magnitude_dbg_batch(self, use_adv_loss: bool,  step: int = None) -> (LossesPRFeatures, LossesPRFeatures):
        if not step:
            step = self._global_step
        with torch.no_grad():
            inferred_batch_tr = self.forward_magnitude_encoder(self.data_tr_batch, eval_mode=False)
            inferred_batch_ts = self.forward_magnitude_encoder(self.data_ts_batch, eval_mode=False)

            tr_losses = self._encoder_losses(self.data_tr_batch, inferred_batch_tr, use_adv_loss=use_adv_loss)
            ts_losses = self._encoder_losses(self.data_ts_batch, inferred_batch_ts, use_adv_loss=use_adv_loss)

            recon_data_tr_batch, recon_data_ts_batch = self.get_dbg_data(inferred_batch_tr, inferred_batch_ts)

            features_recon_tr = self.prepare_dbg_batch(inferred_batch_tr.feature_recon[:self.config.dbg_features_batch],
                                                        grid_ch=True)
            features_recon_ts = self.prepare_dbg_batch(inferred_batch_ts.feature_recon[:self.config.dbg_features_batch],
                                                       grid_ch=True)

            features_enc_tr = self.prepare_dbg_batch(inferred_batch_tr.feature_encoder[:self.config.dbg_features_batch],
                                                        grid_ch=True)
            features_enc_ts = self.prepare_dbg_batch(inferred_batch_ts.feature_encoder[:self.config.dbg_features_batch],
                                                       grid_ch=True)

            self._log_images(recon_data_tr_batch, tag_name='train_en_magnitude/recon', step=step)
            self._log_images(recon_data_ts_batch, tag_name='test_en_magnitude/recon', step=step)

            self._log_images(features_recon_tr, tag_name='train_en_magnitude/features_recon', step=step)
            self._log_images(features_recon_ts, tag_name='test_en_magnitude/features_recon', step=step)

            self._log_images(features_enc_tr, tag_name='train_en_magnitude/features_encoder', step=step)
            self._log_images(features_enc_ts, tag_name='test_en_magnitude/features_encoder', step=step)
        return tr_losses, ts_losses

    def _save_gan_models(self, step: int) -> None:
        if self.save_path is not None:
            save_state = {'ae_model': self.ae_net.state_dict(),
                          'magnitude_encoder': self.phase_predictor.state_dict(),
                          'opt_magnitude_encoder': self.optimizer_en.state_dict()}
            if self.config.use_gan:
                save_state['img_discriminator'] = self.img_discriminator.state_dict()
                save_state['opt_discr'] = self.optimizer_discr.state_dict()
                if self.config.predict_out == 'features':
                    save_state['features_discriminator'] = self.features_discriminator.state_dict()
            if self.config.is_train_ae:
                save_state['opt_ae'] = self.optimizer_ae.state_dict()
            torch.save(save_state, os.path.join(self.save_path, f'phase-retrieval-gan-model.pt'))

    def train_epoch_ae(self, epoch: int = 0) -> LossesPRFeatures:
        train_losses = []
        p_bar_train_data_loader = tqdm(self.train_loader)
        for batch_idx, (data_batch, labels_batch) in enumerate(p_bar_train_data_loader):
            data_batch = data_batch.to(self.device)

            self.optimizer_ae.zero_grad()

            inferred_batch = self.forward_ae(data_batch)
            ae_losses = self._ae_losses(data_batch, inferred_batch)

            ae_losses.total.backward()
            self.optimizer_ae.step()
            train_losses.append(ae_losses.detach())

            if batch_idx % self.log_interval == 0:
                self._log.info(f'Train Epoch: {epoch} [{batch_idx * len(data_batch)}/{len(self.train_loader.dataset)} '
                                f'({100. * batch_idx / len(self.train_loader):.0f}%)], ae_losses: {ae_losses}')
                self._add_losses_tensorboard('ae/train', ae_losses, self._global_step)

            self._global_step += 1

        train_losses = LossesPRFeatures.merge(train_losses)
        return train_losses

    def fit(self, data_batch: Tensor, lr: float, lr_milestones: List[int], lr_reduce_rate: float, n_iter: int, name: str) -> (InferredBatch, LossesPRFeatures):
        self._global_step = 0
        init_params_state_state = copy.deepcopy(self.phase_predictor.state_dict())
        fit_optimizer = optim.Adam(params=self.phase_predictor.parameters(), lr=lr)
        lr_scheduler_fitter = MultiStepLR(fit_optimizer, lr_milestones, lr_reduce_rate)

        real_labels = torch.ones((data_batch.shape[0], 1), device=self.device, dtype=data_batch.dtype)
        losses_fit = []
        for ind_iter in tqdm(range(n_iter)):
            fit_optimizer.zero_grad()
            inferred_batch = self.forward_magnitude_encoder(data_batch)
            fft_magnitude_recon = self.forward_magnitude_fft(inferred_batch.img_recon)

            l2_magnitude_loss = self.l2_loss(inferred_batch.fft_magnitude.detach(), fft_magnitude_recon)
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
                l2_grad_encoder_norm, _ = l2_grad_norm(self.phase_predictor)
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
        self.phase_predictor.load_state_dict(init_params_state_state)

        return inferred_batch, losses_fit

    def train_epoch_en_magnitude(self, epoch: int = 0, use_adv_loss: bool = False) -> LossesPRFeatures:
        train_losses = []
        p_bar_train_data_loader = tqdm(self.train_loader)
        for batch_idx, (data_batch, labels_batch) in enumerate(p_bar_train_data_loader):
            data_batch = data_batch.to(self.device)

            inferred_batch, tr_losses = self._train_step_generator(data_batch, use_adv_loss=use_adv_loss)

            if self.config.use_gan:
                self._train_step_discriminator(data_batch, inferred_batch, tr_losses)

            train_losses.append(tr_losses.detach())

            if batch_idx % self.log_interval == 0:
                l2_grad_magnitude_encoder_norm, _ = l2_grad_norm(self.phase_predictor)
                grad_losses = LossesGradNorms(l2_grad_magnitude_encoder=l2_grad_magnitude_encoder_norm)
                if self.config.use_gan:
                    grad_losses.l2_grad_img_discriminator, _ = l2_grad_norm(self.img_discriminator)
                    if self.config.predict_out == 'features':
                        grad_losses.l2_grad_features_discriminator, _ = l2_grad_norm(self.features_discriminator)

                self._log.info(f'Train Epoch: {epoch} [{batch_idx * len(data_batch)}/{len(self.train_loader.dataset)}'
                                f'({100. * batch_idx / len(self.train_loader):.0f}%)], {train_losses[batch_idx]}, '
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

    def _discrim_ls_loss(self, discriminator: nn.Module, real_img: Tensor, generated_img: Tensor,
                         real_labels: Tensor, fake_labels: Tensor) -> Tensor:
        real_loss = self.adv_loss(discriminator(real_img.detach()).validity, real_labels)
        fake_loss = self.adv_loss(discriminator(generated_img.detach()).validity, fake_labels)
        disrm_loss = 0.5 * (real_loss + fake_loss)
        return disrm_loss

    def _train_step_discriminator(self, data_batch: Tensor, inferred_batch: InferredBatch, tr_losses: LossesPRFeatures):
        self.optimizer_discr.zero_grad()
        # Measure discriminator's ability to classify real from generated samples
        batch_size = data_batch.shape[0]
        real_labels = torch.ones((batch_size, 1), device=self.device, dtype=data_batch.dtype)
        fake_labels = torch.zeros((batch_size, 1), device=self.device, dtype=data_batch.dtype)
        if self.config.predict_out == 'features':
            feature_disrm_loss = self._discrim_ls_loss(self.features_discriminator,
                                                       real_img=inferred_batch.feature_encoder,
                                                       generated_img=inferred_batch.feature_recon,
                                                       real_labels=real_labels,
                                                       fake_labels=fake_labels)
            tr_losses.disrm_loss = self.config.lambda_discrim_features * feature_disrm_loss
        else:
            feature_disrm_loss = None
        img_disrm_loss = self._discrim_ls_loss(self.img_discriminator,
                                               real_img=inferred_batch.decoded_img,
                                               generated_img=inferred_batch.img_recon,
                                               real_labels=real_labels,
                                               fake_labels=fake_labels)
        tr_losses.disrm_loss = self.config.lambda_discrim_img * img_disrm_loss

        tr_losses.img_disrm_loss = img_disrm_loss
        tr_losses.features_disrm_loss = feature_disrm_loss
        tr_losses.disrm_loss.backward()
        if self.config.clip_discriminator_grad:
            torch.nn.utils.clip_grad_norm_(self.img_discriminator.parameters(),
                                           self.config.clip_discriminator_grad)
        self.optimizer_discr.step()

    def _train_step_generator(self, data_batch: Tensor,
                              use_adv_loss: bool = False) -> (InferredBatch, LossesPRFeatures):
        self.optimizer_en.zero_grad()
        inferred_batch = self.forward_magnitude_encoder(data_batch)
        tr_losses = self._encoder_losses(data_batch, inferred_batch, use_adv_loss=use_adv_loss)
        tr_losses.total.backward()
        if self.config.clip_encoder_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.phase_predictor.parameters(), self.config.clip_encoder_grad)
        self.optimizer_en.step()

        return inferred_batch, tr_losses

    def _ae_losses(self, data_batch: Tensor, inferred_batch: InferredBatch) -> LossesPRFeatures:
        fft_magnitude_recon = self.forward_magnitude_fft(inferred_batch.img_recon)
        l2_img_loss = self.l2_loss(data_batch, inferred_batch.img_recon)
        l1_sparsity_features = torch.mean(inferred_batch.feature_recon.abs())
        l2_magnitude_loss = self.l2_loss(inferred_batch.fft_magnitude.detach(), fft_magnitude_recon)
        l2_features_loss = self.l2_loss(inferred_batch.feature_encoder, inferred_batch.feature_recon)
        total_loss = l2_img_loss
        losses = LossesPRFeatures(total=total_loss,
                                  l2_img=l2_img_loss,
                                  l2_features=l2_features_loss,
                                  l2_magnitude=l2_magnitude_loss,
                                  l1_sparsity_features=l1_sparsity_features)
        return losses

    def _encoder_losses(self, data_batch: Tensor,
                        inferred_batch: InferredBatch,
                        use_adv_loss: bool = False) -> LossesPRFeatures:

        fft_magnitude_recon = self.forward_magnitude_fft(inferred_batch.img_recon)
        l2_img_recon_loss = self.l2_loss(data_batch, inferred_batch.img_recon)
        if self.config.predict_out == 'features':
            l2_features_loss = self.l2_loss(inferred_batch.feature_encoder, inferred_batch.feature_recon)
            l1_sparsity_features = torch.mean(inferred_batch.feature_recon.abs())
        else:
            l2_features_loss = None
            l1_sparsity_features = None

        l2_magnitude_loss = self.l2_loss(inferred_batch.fft_magnitude.detach(), fft_magnitude_recon)
        l2_features_realness = 0.5 * torch.mean(torch.square(inferred_batch.intermediate_features.imag.abs()))

        real_labels = torch.ones((data_batch.shape[0], 1), device=self.device, dtype=data_batch.dtype)

        total_loss = self.config.lambda_img_recon_loss * l2_img_recon_loss
        if use_adv_loss:
            img_discrim_batch: DiscriminatorBatch = self.img_discriminator(inferred_batch.img_recon)
            img_adv_loss = self.adv_loss(img_discrim_batch.validity, real_labels)
            total_loss += self.config.lambda_img_adv_loss * img_adv_loss

            if self.config.predict_out == 'features':
                f_disc_generated_batch: DiscriminatorBatch = self.features_discriminator(inferred_batch.feature_recon)
                f_disc_real_batch: DiscriminatorBatch = self.features_discriminator(inferred_batch.feature_encoder)
                p_loss_discrim_f = l2_perceptual_loss(f_disc_generated_batch.features, f_disc_real_batch.features,
                                                      weights=[1, 0])
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

        total_loss += self.config.lambda_magnitude_recon_loss * l2_magnitude_loss + \
                      self.config.lambda_features_realness * l2_features_realness

        if self.config.predict_out == 'features':
            total_loss += self.config.lambda_features_recon_loss * l2_features_loss + \
                          self.config.lambda_sparsity_features * l1_sparsity_features

        losses = LossesPRFeatures(total=total_loss,
                                  l2_img=l2_img_recon_loss,
                                  l2_features=l2_features_loss,
                                  l2_magnitude=l2_magnitude_loss,
                                  l1_sparsity_features=l1_sparsity_features,
                                  realness_features=l2_features_realness,
                                  img_adv_loss=img_adv_loss,
                                  features_adv_loss=features_adv_loss,
                                  perceptual_disrim_features=p_loss_discrim_f)

        return losses

    def test_eval_ae(self) -> LossesPRFeatures:
        losses = []
        self.set_eval_mode()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data_batch = data.to(self.device)
                recon_batch, _ = self.ae_net(data_batch)
                l2_img_loss = self.l2_loss(data_batch, recon_batch)
                losses.append(l2_img_loss)
            losses = torch.stack(losses).mean()
        self.set_train_model()
        return losses

    def test_eval_en_magnitude(self, use_adv_loss: bool = False) -> LossesPRFeatures:
        losses_ts = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data_batch = data.to(self.device)
                inferred_batch = self.forward_magnitude_encoder(data_batch, eval_mode=False)
                losses_ts_ = self._encoder_losses(data_batch, inferred_batch, use_adv_loss=use_adv_loss)
                losses_ts.append(losses_ts_.detach())

            losses_ts = LossesPRFeatures.merge(losses_ts).mean()
            return losses_ts

    def train_ae(self):
        self._global_step = 0
        init_ae_err = self.test_eval_ae()
        self._log.info(f' AE training: init ts err: {init_ae_err}')
        self._log_ae_train_dbg_images(0)
        self._add_losses_tensorboard('ae/test', init_ae_err.mean(), self._global_step)
        for epoch in range(1, self.n_epochs_ae + 1):
            train_losses = self.train_epoch_ae(epoch)
            train_losses = train_losses.mean()

            ts_eval_ae_losses = self.test_eval_ae()
            ts_eval_ae_losses.lr = torch.tensor(self._lr_scheduler_ae.get_last_lr()[0])

            self._lr_scheduler_ae.step()
            self._add_losses_tensorboard('ae/test', ts_eval_ae_losses, self._global_step)
            self._log.info(f' AE training: Epoch {epoch}, l2_recon_err_tr: {train_losses}, '
                            f'l2_recon_err_ts: {ts_eval_ae_losses}')
            with torch.no_grad():
                self._log_ae_train_dbg_images(epoch)

                if self.save_path is not None:
                    ae_state = {'ae_model': self.ae_net.state_dict(),
                                'opt_ae': self.optimizer_ae.state_dict()}
                    torch.save(ae_state, os.path.join(self.save_path, f'ae_model.pt'))

    def _log_ae_train_dbg_images(self, step: Optional[int] = None) -> None:
        if not step:
            step = self._global_step
        with torch.no_grad():
            recon_data_tr_batch = self.forward_ae(self.data_tr_batch, eval_mode=False)
            recon_data_ts_batch = self.forward_ae(self.data_ts_batch, eval_mode=False)
            recon_tr_batch_dbg = self.get_dbg_batch(self.data_tr_batch, recon_data_tr_batch)
            recon_ts_batch_dbg = self.get_dbg_batch(self.data_ts_batch, recon_data_ts_batch)
            features_tr = self.prepare_dbg_batch(recon_data_tr_batch.feature_recon[:self.config.dbg_features_batch],
                                                 grid_ch=True)
            features_ts = self.prepare_dbg_batch(recon_data_ts_batch.feature_recon[:self.config.dbg_features_batch],
                                                 grid_ch=True)
            self._log_images(recon_tr_batch_dbg, tag_name='train-ae/recon', step=step)
            self._log_images(recon_ts_batch_dbg, tag_name='test-ae/recon', step=step)
            self._log_images(features_tr, tag_name='train-ae/features', step=step)
            self._log_images(features_ts, tag_name='test-ae/features', step=step)

    def get_dbg_data(self, inferred_batch_tr: InferredBatch,
                     inferred_batch_ts: InferredBatch) -> (np.ndarray, np.ndarray):

        dbg_data_tr_batch = self.get_dbg_batch(self.data_tr_batch, inferred_batch_tr)
        dbg_data_ts_batch = self.get_dbg_batch(self.data_ts_batch, inferred_batch_ts)

        return dbg_data_tr_batch, dbg_data_ts_batch

    def get_dbg_batch(self, data_batch: Tensor, inferred_batch: InferredBatch) -> np.ndarray:
        data_batch = self.prepare_dbg_batch(data_batch)
        recon_batch = self.prepare_dbg_batch(inferred_batch.img_recon)
        dbg_data_batch = [data_batch]
        if inferred_batch.decoded_img is not None:
            dbg_data_batch.append(self.prepare_dbg_batch(inferred_batch.decoded_img))
        dbg_data_batch.append(recon_batch)
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


if __name__ == '__main__':
    fire.Fire(run_ae_features_trainer)
