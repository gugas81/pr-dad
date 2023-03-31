import torch
import numpy as np
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import fire
import os

from typing import Optional, List, Dict, Union
from tqdm import tqdm
from models import Discriminator
from torch.optim.lr_scheduler import MultiStepLR

from common import LossesPRFeatures, InferredBatch, ConfigTrainer,  LossesGradNorms,  DiscriminatorBatch
from common import im_concatenate, DataBatch

from training.base_phase_retrieval_trainer import BaseTrainerPhaseRetrieval
from training.phase_retrieval_model import PhaseRetrievalAeModel
from training.utils import ModulesNames
from training.phase_retrival_evaluator import Evaluator
import models.losses as los_fun


class TrainerPhaseRetrievalAeFeatures(BaseTrainerPhaseRetrieval):
    def __init__(self, config: ConfigTrainer, experiment_name: str):
        super(TrainerPhaseRetrievalAeFeatures, self).__init__(config=config, experiment_name=experiment_name)

        self.adv_loss = nn.MSELoss()

        self.l2_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

        self.l2_img_loss = los_fun.LossImg(loss_type='l2', rot180=config.loss_rot180, device=self.device)
        self.l1_img_loss = los_fun.LossImg(loss_type='l1', rot180=config.loss_rot180, device=self.device)

        if self._config.ae_type == 'wavelet-net':
            self.l2_f_loss = los_fun.DwtCoeffLoss(n_subbands=self._config.n_features,
                                                  loss_type='l2',
                                                  rot180=config.loss_rot180,
                                                  device=self.device)
            self.l1_f_loss = los_fun.DwtCoeffLoss(n_subbands=self._config.n_features,
                                                  loss_type='l1',
                                                  rot180=config.loss_rot180,
                                                  device=self.device)
        else:
            self.l2_f_loss = los_fun.LossImg(loss_type='l2', rot180=config.loss_rot180, device=self.device)
            self.l1_f_loss = los_fun.LossImg(loss_type='l1', rot180=config.loss_rot180, device=self.device)

        self.sparsity_f_loss = los_fun.SparsityL1Loss(dc_comp=(self._config.ae_type == 'wavelet-net'))

        if self._config.use_lpips:
            self._lpips_loss = los_fun.LossImg(loss_type='lpips', rot180=config.loss_rot180, device=self.device)
        else:
            self._lpips_loss = None

        self.n_epochs_ae = config.n_epochs_ae
        if self._config.use_amp:
            self._scaler = torch.cuda.amp.GradScaler()
        else:
            self._scaler = None

        self._generator_model = PhaseRetrievalAeModel(config=self._config, s3=self._s3, log=self._log)

        self._init_discriminators()

        self._generator_model.set_train_mode(ae_train=self._config.ae_decoder_fine_tune)
        self._generator_model.set_device(self.device)

        self.optimizers_generator = {}

        if self._config.is_train_ae and (self._generator_model.ae_net is not None) and \
                (ModulesNames.ae_model not in self._config.optim_exclude):

            self.optimizer_ae = optim.Adam(params=self._generator_model.ae_net.parameters(), lr=self._config.lr_ae)

            if self._config.use_ae_dictionary:
                self.optimizer_dict = optim.Adam(params=self._generator_model.ae_net.dictionary.parameters(),
                                                 lr=self._config.lr_dict)
        else:
            self.optimizer_ae = None
            self.optimizer_dict = None

        if ModulesNames.magnitude_encoder not in self._config.optim_exclude:
            self.optimizer_en = optim.Adam(params=self._generator_model.phase_predictor.parameters(),
                                           lr=self._config.lr_enc)
            self.optimizers_generator.update({ModulesNames.opt_magnitude_enc: self.optimizer_en})
        else:
            self.optimizer_en = None
        if self._config.use_ref_net and (ModulesNames.ref_net not in self._config.optim_exclude):
            self.optimizer_ref_net = optim.Adam(params=self._generator_model.ref_unet.parameters(),
                                                lr=self._config.lr_ref_net)
            self.optimizers_generator.update({ModulesNames.opt_ref_net: self.optimizer_ref_net})
        else:
            self.optimizer_ref_net = None

        if self._config.predict_out == 'features' and self._config.ae_decoder_fine_tune:
            decoder_params = self._generator_model.ae_net.get_submodule('_decoder').parameters()
            optimizer_ae_decoder = optim.Adam(params=decoder_params,  lr=self._config.lr_ae_decoder)
            self.optimizers_generator.update({ModulesNames.opt_ae_decoder: optimizer_ae_decoder})

        if self._config.use_gan:
            if self._config.predict_out == 'features':
                self.features_discriminator.train()
                self.features_discriminator.to(device=self.device)

            self.img_discriminator.train()
            self.img_discriminator.to(device=self.device)
            disrim_params = list(self.img_discriminator.parameters())
            if self._config.predict_out == 'features':
                disrim_params += list(self.features_discriminator.parameters())
            self.optimizer_discr = optim.Adam(params=disrim_params, lr=self._config.lr_discr)

        else:
            self.optimizer_discr = None

        self._lr_schedulers = {}
        if self.optimizer_en:
            self._lr_schedulers.update({ModulesNames.opt_magnitude_enc: MultiStepLR(self.optimizer_en,
                                                                                    self._config.lr_milestones_en,
                                                                                    self._config.lr_reduce_rate_en)})

        if self.optimizer_discr:
            self._lr_schedulers.update({ModulesNames.opt_discriminators: MultiStepLR(self.optimizer_discr,
                                                                                     self._config.lr_milestones_en,
                                                                                     self._config.lr_reduce_rate_en)})

        if self.optimizer_ref_net:
            self._lr_schedulers.update({ModulesNames.opt_ref_net: MultiStepLR(self.optimizer_ref_net,
                                                                              self._config.lr_milestones_en,
                                                                              self._config.lr_reduce_rate_en)})

        if self._config.is_train_ae:
            # self._lr_schedulers.update({ModulesNames.opt_ae: MultiStepLR(self.optimizer_ae,
            #                                                              self._config.lr_milestones_en,
            #                                                              self._config.lr_reduce_rate_en)})
            self._lr_scheduler_ae = MultiStepLR(self.optimizer_ae,
                                                self._config.lr_milestones_en,
                                                self._config.lr_reduce_rate_en)

            if self._config.use_ae_dictionary:
                self._lr_scheduler_dict = MultiStepLR(self.optimizer_dict,
                                                      self._config.lr_milestones_ae,
                                                      self._config.lr_reduce_rate_ae)

        else:
            self._lr_scheduler_ae = None

        self.load_models()

    def _init_discriminators(self):
        if self._config.use_gan:
            if self._config.predict_out == 'features':
                in_conv_ch = self._generator_model.ae_net.n_enc_features_ch \
                    if self._config.discrim_features_ch is None else self._config.discrim_features_ch
                self.features_discriminator = Discriminator(input_ch=self._generator_model.ae_net.n_enc_features_ch,
                                                            in_conv_ch=in_conv_ch,
                                                            input_norm_type=self._config.disrim_input_norm,
                                                            fc_norm_type=self._config.disrim_fc_norm,
                                                            img_size=self._generator_model.ae_net.n_features_size,
                                                            n_fc_layers=self._config.disrim_features_fc_layers,
                                                            deep_conv_net=1,
                                                            reduce_validity=True,
                                                            use_res_blocks=False,
                                                            active_type=self._config.activation_discrim)
                self._log.debug(f'Features Discriminator \n {self.features_discriminator}')
            else:
                self.features_discriminator = None

            self.img_discriminator = Discriminator(in_conv_ch=self._config.disrim_in_conv_ch,
                                                   input_norm_type=self._config.disrim_input_norm,
                                                   fc_norm_type=self._config.disrim_fc_norm,
                                                   img_size=self.img_size,
                                                   n_fc_layers=self._config.disrim_fc_layers,
                                                   deep_conv_net=self._config.deep_ae,
                                                   reduce_validity=True,
                                                   active_type=self._config.activation_discrim)
            self._log.debug(f'Image Discriminator \n {self.img_discriminator}')
        else:
            self.img_discriminator = None
            self.features_discriminator = None

    def train(self) -> (LossesPRFeatures, LossesPRFeatures, LossesPRFeatures):
        train_en_losses, test_en_losses, test_ae_losses = [], [], []
        if self._config.debug_mode:
            losses_dbg_batch_tr, losses_dbg_batch_ts = self._log_en_magnitude_dbg_batch(self._config.use_gan,
                                                                                        self._global_step)

        if self._config.is_train_ae:
            assert self._config.predict_out == 'features'
            train_ae_losses, test_ae_losses = self.train_ae()

        if self._config.is_train_encoder:
            if self._config.predict_out == 'features':
                self._generator_model.ae_net.eval()
                test_ae_losses = self.test_eval_ae()
                self._log.info(f' AE training: ts err: {test_ae_losses.mean()}')
            train_en_losses, test_en_losses = self.train_en_magnitude()

        return train_en_losses, test_en_losses, test_ae_losses

    def train_en_magnitude(self) -> (LossesPRFeatures, LossesPRFeatures):
        self._global_step = 0
        use_adv_loss = self._config.use_gan  # and epoch > 5
        train_en_losses, test_en_losses = [], []
        self._log.info(f'train_en_magnitude is Staring')
        init_ts_losses = self.test_eval_en_magnitude(use_adv_loss)
        self._add_losses_tensorboard('en-magnitude/test', init_ts_losses, self._global_step)
        evaluator = Evaluator(model_type=self._generator_model, data_holder=self._data_holder)
        eval_test_df = self._add_metrics_evaluator_test(evaluator, self._global_step)
        self._log.debug(f'INIT  -- Evaluation test data \n {eval_test_df}')
        for epoch in range(1, self.n_epochs + 1):
            self._log.info(f'Train Epoch{epoch}')

            tr_losses = self.train_epoch_en_magnitude(epoch, use_adv_loss)
            ts_losses = self.test_eval_en_magnitude(use_adv_loss)
            ts_losses.lr = self.get_last_lr_enc()
            self._add_losses_tensorboard('en-magnitude/test', ts_losses, self._global_step)

            for name_optim, lr_scheduler in self._lr_schedulers.items():
                lr_scheduler.step()

            train_en_losses.append(tr_losses)
            test_en_losses.append(ts_losses)

            self._log.info(f'Magnitude Encoder Epoch={epoch}, '
                            f'Train Losses: {tr_losses}, '
                            f'Test Losses: {ts_losses} ')
            if not self._config.debug_mode:
                self._save_gan_models(epoch, force=(epoch == self.n_epochs))
            self._generator_model.set_eval_mode()
            losses_dbg_batch_tr, losses_dbg_batch_ts = self._log_en_magnitude_dbg_batch(use_adv_loss, self._global_step)
            self._add_losses_tensorboard('dbg-batch-en-magnitude/train', losses_dbg_batch_tr, self._global_step)
            self._add_losses_tensorboard('dbg-batch-en-magnitude/test', losses_dbg_batch_ts, self._global_step)
            eval_test_df = self._add_metrics_evaluator_test(evaluator, self._global_step)
            self._log.debug(f'Epoch:{epoch} -- Evaluation test data \n {eval_test_df}')

        train_en_losses = LossesPRFeatures.merge(train_en_losses)
        test_en_losses = LossesPRFeatures.merge(test_en_losses)

        return train_en_losses, test_en_losses

    def get_last_lr_enc(self) -> Dict[str, Tensor]:
        return {name_optim: torch.tensor(lr_scheduler.get_last_lr()[0])
                for name_optim, lr_scheduler in self._lr_schedulers.items()}

        # torch.tensor(self._lr_schedulers[0].get_last_lr()[0])

    def _add_metrics_evaluator_test(self, evaluator: Evaluator, step: int = None) -> pd.DataFrame:
        eval_test_df: pd.DataFrame = evaluator.benchmark_dataset(type_ds='test')
        for metric_type in [evaluator.RECON_REF, evaluator.RECON]:
            recon_eval = eval_test_df.loc[metric_type]
            for metric_name, row in recon_eval.iterrows():
                mean_val = row[evaluator.MEAN]
                self._tensorboard.add_scalar(f"{metric_name}/eval-{metric_type}-{evaluator.RECON}", mean_val, step)
        return eval_test_df

    def train_epoch_ae(self, epoch: int = 0, train_dict: bool = False) -> LossesPRFeatures:
        train_losses = []
        p_bar_train_data_loader = tqdm(self._data_holder.train_paired_loader)
        self._generator_model.set_train_mode(ae_train=self._config.ae_decoder_fine_tune)
        for batch_idx, data_batch in enumerate(p_bar_train_data_loader):
            data_batch = self.prepare_data_batch(data_batch, is_train=True)
            if train_dict:
                self.optimizer_dict.zero_grad()
            else:
                self.optimizer_ae.zero_grad()
            inferred_batch = self._generator_model.forward_ae(data_batch)
            ae_losses = self._ae_losses(data_batch, inferred_batch)

            ae_losses.total.backward()
            if train_dict:
                self.optimizer_dict.step()
            else:
                self.optimizer_ae.step()
            train_losses.append(ae_losses.detach())

            if batch_idx % self.log_interval == 0:
                self._log.info(f'Train Epoch: {epoch}, train_dict: {train_dict}, '
                               f'[{batch_idx * len(data_batch)}/{len(self._data_holder.train_paired_loader)} '
                                f'({100. * batch_idx / len(self._data_holder.train_paired_loader):.0f}%)], ae_losses: {ae_losses}')
                self._add_losses_tensorboard('ae/train', ae_losses, self._global_step)

            if batch_idx % self._config.log_image_interval == 0:
                with torch.no_grad():
                    self._generator_model.set_eval_mode()
                    self._log_ae_train_dbg_batch(self._global_step)
                    self._generator_model.set_train_mode(ae_train=self._config.ae_decoder_fine_tune)

            self._global_step += 1

        train_losses = LossesPRFeatures.merge(train_losses)
        return train_losses

    def train_epoch_en_magnitude(self, epoch: int = 0, use_adv_loss: bool = False) -> LossesPRFeatures:
        train_losses = []
        p_bar_train_data_loader = tqdm(self._data_holder.train_paired_loader)
        self._generator_model.set_train_mode(ae_train=self._config.ae_decoder_fine_tune)
        for batch_idx, data_batch in enumerate(p_bar_train_data_loader):
            data_batch = self.prepare_data_batch(data_batch, is_train=True)
            inferred_batch, tr_losses = self._train_step_generator(data_batch, use_adv_loss=use_adv_loss)

            if self._config.use_gan:
                self._train_step_discriminator(data_batch, inferred_batch, tr_losses)

            train_losses.append(tr_losses.detach())

            if batch_idx % self.log_interval == 0:

                l2_grad_magnitude_encoder_norm, _ = los_fun.l2_grad_norm(self._generator_model.phase_predictor)
                grad_losses = LossesGradNorms(l2_grad_magnitude_encoder=l2_grad_magnitude_encoder_norm)
                if self._config.use_gan:
                    grad_losses.l2_grad_img_discriminator, _ = los_fun.l2_grad_norm(self.img_discriminator)
                    if self._config.predict_out == 'features':
                        grad_losses.l2_grad_features_discriminator, _ = los_fun.l2_grad_norm(self.features_discriminator)
                self._log.info(f'Train Epoch: {epoch} [{batch_idx * len(data_batch)}/{len(self._data_holder.train_paired_loader.dataset)}'
                               f'({100. * batch_idx / len(self._data_holder.train_paired_loader):.0f}%)], '
                               f'{train_losses[batch_idx]}, {grad_losses}')
                self._add_losses_tensorboard('en-magnitude/train', tr_losses, self._global_step)
                self._add_losses_tensorboard('en-magnitude/train', grad_losses, self._global_step)

            if batch_idx % self._config.log_image_interval == 0:
                self._generator_model.set_eval_mode()
                losses_dbg_batch_tr, losses_dbg_batch_ts = self._log_en_magnitude_dbg_batch(use_adv_loss,
                                                                                            self._global_step)
                self._add_losses_tensorboard('dbg-batch-en-magnitude/train', losses_dbg_batch_tr, self._global_step)
                self._add_losses_tensorboard('dbg-batch-en-magnitude/test', losses_dbg_batch_ts, self._global_step)
                self._generator_model.set_train_mode(ae_train=self._config.ae_decoder_fine_tune)

            self._global_step += 1

        train_losses = LossesPRFeatures.merge(train_losses).mean()
        return train_losses

    def test_eval_ae(self) -> LossesPRFeatures:
        ae_losses = []
        self._generator_model.set_eval_mode()
        with torch.no_grad():
            len_data = len(self._data_holder.test_loader)
            data_iter = iter(self._data_holder.test_loader)
            for batch_idx in tqdm(range(len_data)):
                try:
                    data_batch = next(data_iter)
                except Exception as e:
                    self._log.error(f'Error loading item {batch_idx} from the dataset: {e}')

                data_batch = self.prepare_data_batch(data_batch, is_train=False)
                inferred_batch = self._generator_model.forward_ae(data_batch)
                ae_losses_batch = self._ae_losses(data_batch, inferred_batch)
                ae_losses.append(ae_losses_batch)
            ae_losses = LossesPRFeatures.merge(ae_losses)
        self._generator_model.set_train_mode(ae_train=self._config.ae_decoder_fine_tune)
        return ae_losses

    def test_eval_en_magnitude(self, use_adv_loss: bool = False) -> LossesPRFeatures:
        losses_ts = []
        self._generator_model.set_eval_mode()
        with torch.no_grad():
            for batch_idx, data_batch in enumerate(self._data_holder.test_loader):
                data_batch = self.prepare_data_batch(data_batch, is_train=False)
                inferred_batch = self._generator_model.forward_magnitude_encoder(data_batch)
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
        self._log_ae_train_dbg_batch(step=0)
        self._add_losses_tensorboard('ae/test', init_ae_losses, self._global_step)
        n_epochs = self.n_epochs_ae
        if self._config.use_ae_dictionary:
            n_epochs *= 2
        for epoch in range(1, n_epochs):
            train_dict = epoch > self.n_epochs_ae
            tr_losses_epoch = self.train_epoch_ae(epoch, train_dict=train_dict)
            tr_losses_epoch = tr_losses_epoch.mean()

            ts_losses_epoch = self.test_eval_ae()

            if train_dict:
                ts_losses_epoch.lr = torch.tensor(self._lr_scheduler_dict.get_last_lr()[0])
                self._lr_scheduler_dict.step()
            else:
                ts_losses_epoch.lr = torch.tensor(self._lr_scheduler_ae.get_last_lr()[0])
                self._lr_scheduler_ae.step()

            self._add_losses_tensorboard('ae/test', ts_losses_epoch, self._global_step)
            self._log.info(f'AE training: Epoch {epoch}, train_dict: {train_dict}, '
                           f'l2_recon_err_tr: {tr_losses_epoch}, '
                           f'l2_recon_err_ts: {ts_losses_epoch}')
            with torch.no_grad():
                self._log_ae_train_dbg_batch(self._global_step)

                if self.models_path is not None:
                    ae_state = {ModulesNames.config: self._config.as_dict(),
                                ModulesNames.ae_model: self._generator_model.ae_net.state_dict(),
                                ModulesNames.opt_ae: self.optimizer_ae.state_dict()}
                    torch.save(ae_state, os.path.join(self.models_path, f'ae_model.pt'))
            tr_losses.append(tr_losses_epoch)
            ts_losses.append(ts_losses_epoch)
        tr_losses = LossesPRFeatures.merge(tr_losses)
        ts_losses = LossesPRFeatures.merge(ts_losses)
        return tr_losses, ts_losses

    def load_models(self):
        if self._config.path_pretrained is not None:
            loaded_sate = self.load_state(self._config.path_pretrained)

            self._generator_model.load_modules(loaded_sate)

            if self._config.use_gan:
                self._generator_model.load_module(loaded_sate,
                                                  self.img_discriminator,
                                                  ModulesNames.img_discriminator)
                self._generator_model.load_module(loaded_sate,
                                                  self.optimizer_discr,
                                                  ModulesNames.opt_discriminators)
                if self._config.predict_out == 'features':
                    self._generator_model.load_module(loaded_sate,
                                                      self.features_discriminator,
                                                      ModulesNames.features_discriminator)

            if self._config.is_train_encoder:
                for opt_name, optimizer in self.optimizers_generator.items():
                    self._generator_model.load_module(loaded_sate,
                                                      optimizer,
                                                      opt_name)

            if self._config.is_train_ae:
                self._generator_model.load_module(loaded_sate,
                                                  self.optimizer_ae,
                                                  ModulesNames.opt_ae)

    def _discrim_ls_loss(self, discriminator: nn.Module, real_img: Tensor, generated_imgs: List[Tensor],
                         real_labels: Tensor, fake_labels: Tensor) -> Tensor:
        real_loss = self.adv_loss(discriminator(real_img.detach()).validity, real_labels)
        fake_loss = torch.zeros_like(real_loss)
        for gen_img in generated_imgs:
            fake_loss += self.adv_loss(discriminator(gen_img.detach()).validity, fake_labels)
        disrm_loss = 0.5 * (real_loss + fake_loss)
        return disrm_loss

    def _train_step_discriminator(self, data_batch: DataBatch, inferred_batch: InferredBatch, tr_losses: LossesPRFeatures):
        self.optimizer_discr.zero_grad()
        # Measure discriminator's ability to classify real from generated samples
        batch_size = data_batch.image.shape[0]
        real_labels = torch.ones((batch_size, 1), device=self.device, dtype=data_batch.image.dtype)
        fake_labels = torch.zeros((batch_size, 1), device=self.device, dtype=data_batch.image.dtype)

        # generated_img = [inferred_batch.img_recon]
        # if self._config.use_ref_net:
        #     generated_img.append(inferred_batch.img_recon_ref)

        if self._config.predict_out == 'features':
            real_img = inferred_batch.decoded_img
        else:
            real_img = data_batch.image

        img_disrm_loss = self._discrim_ls_loss(self.img_discriminator,
                                               real_img=real_img,
                                               generated_imgs=[inferred_batch.img_recon],
                                               real_labels=real_labels,
                                               fake_labels=fake_labels)
        tr_losses.disrm_loss = self._config.lambda_discrim_img * img_disrm_loss

        if self._config.use_ref_net:
            ref_img_disrm_loss = self._discrim_ls_loss(self.img_discriminator,
                                                       real_img=real_img,
                                                       generated_imgs=[inferred_batch.img_recon_ref],
                                                       real_labels=real_labels,
                                                       fake_labels=fake_labels)
            tr_losses.disrm_loss += self._config.lambda_discrim_ref_img * ref_img_disrm_loss
        else:
            ref_img_disrm_loss = None

        if self._config.predict_out == 'features':
            feature_disrm_loss = self._discrim_ls_loss(self.features_discriminator,
                                                       real_img=inferred_batch.feature_decoder,
                                                       generated_imgs=[inferred_batch.feature_recon_decoder],
                                                       real_labels=real_labels,
                                                       fake_labels=fake_labels)
            tr_losses.disrm_loss += self._config.lambda_discrim_features * feature_disrm_loss
        else:
            feature_disrm_loss = None

        tr_losses.img_disrm_loss = img_disrm_loss
        tr_losses.ref_img_disrm_loss = ref_img_disrm_loss
        tr_losses.features_disrm_loss = feature_disrm_loss
        tr_losses.disrm_loss.backward()
        if self._config.clip_discriminator_grad:
            torch.nn.utils.clip_grad_norm_(self.img_discriminator.parameters(),
                                           self._config.clip_discriminator_grad)
        self.optimizer_discr.step()

    def _train_step_generator(self, data_batch: DataBatch,
                              use_adv_loss: bool = False) -> (InferredBatch, LossesPRFeatures):
        self._zero_grad_optimizers_gen()

        # with torch.cuda.amp.autocast(enabled=self._config.use_amp):
        inferred_batch = self._generator_model.forward_magnitude_encoder(data_batch)
        tr_losses = self._encoder_losses(data_batch, inferred_batch, use_adv_loss=use_adv_loss)
        if self._config.use_amp:
            self._scaler.scale(tr_losses).backward()
        else:
            tr_losses.total.backward()
        if self._config.clip_encoder_grad is not None:
            torch.nn.utils.clip_grad_norm_(self._generator_model.phase_predictor.parameters(),
                                           self._config.clip_encoder_grad)

        self._step_optimizers()

        return inferred_batch, tr_losses

    def _zero_grad_optimizers_gen(self):
        for opt_name, optim_ in self.optimizers_generator.items():
            optim_.zero_grad()

    def _step_optimizers(self):
        for opt_name, optim_ in self.optimizers_generator.items():
            try:
                if self._config.use_amp:
                    self._scaler.step(optim_)
                else:
                    optim_.step()
            except Exception as e:
                self._log.error(f'Run optimizer: {opt_name}, \n error {e}')
                raise RuntimeError(e)

    def _ae_losses(self, data_batch: DataBatch, inferred_batch: InferredBatch) -> LossesPRFeatures:
        fft_magnitude_recon = self._generator_model.forward_magnitude_fft(inferred_batch.img_recon)
        l1_img_loss = self.l1_loss(data_batch.image, inferred_batch.img_recon)
        l2_img_loss = self.l2_loss(data_batch.image, inferred_batch.img_recon)
        l1_sparsity_features = self.sparsity_f_loss(inferred_batch.feature_recon_decoder)
        if self._config.use_ae_dictionary:
            l1_sparsity_dict_coeff = self.sparsity_f_loss(inferred_batch.dict_coeff_encoder)
        else:
            l1_sparsity_dict_coeff = None
        l1_magnitude_loss = self.l1_loss(data_batch.fft_magnitude.detach(), fft_magnitude_recon)
        l2_magnitude_loss = self.l2_loss(data_batch.fft_magnitude.detach(), fft_magnitude_recon)
        l1_features_loss = self.l1_loss(inferred_batch.feature_decoder, inferred_batch.feature_recon_decoder)
        l2_features_loss = self.l2_loss(inferred_batch.feature_decoder, inferred_batch.feature_recon_decoder)
        total_loss = l1_img_loss + l2_img_loss + l2_features_loss + \
                      0.5 * self._config.lambda_sparsity_features * l1_sparsity_features

        if self._config.use_ae_dictionary:
            total_loss += self._config.lambda_sparsity_dict_coeff * l1_sparsity_dict_coeff

        losses = LossesPRFeatures(total=total_loss,
                                  l1_img=l1_img_loss,
                                  l2_img=l2_img_loss,
                                  l1_features=l1_features_loss,
                                  l2_features=l2_features_loss,
                                  l1_magnitude=l1_magnitude_loss,
                                  l2_magnitude=l2_magnitude_loss,
                                  l1_sparsity_features=l1_sparsity_features,
                                  l1_sparsity_dict_coeff=l1_sparsity_dict_coeff)

        self._recon_statistics_metrics(inferred_batch, losses)

        return losses

    def _encoder_losses(self, data_batch: DataBatch,
                        inferred_batch: InferredBatch,
                        use_adv_loss: bool = False) -> LossesPRFeatures:

        # is_paired = data_batch.is_paired

        fft_magnitude_recon = self._generator_model.forward_magnitude_fft(inferred_batch.img_recon)
        total_loss = torch.zeros(1, device=self.device)[0]
        l1_img_recon_loss = self.l1_img_loss(data_batch.image, inferred_batch.img_recon)
        l2_img_recon_loss = self.l2_img_loss(data_batch.image, inferred_batch.img_recon)

        if self._config.predict_out == 'features':
            l1_features_loss = self.l1_f_loss(inferred_batch.feature_decoder, inferred_batch.feature_recon_decoder)
            l2_features_loss = self.l2_f_loss(inferred_batch.feature_decoder, inferred_batch.feature_recon_decoder)
            l1_sparsity_features = self.sparsity_f_loss(inferred_batch.feature_recon)
            if self._config.use_ae_dictionary:
                l1_sparsity_dict_coeff = torch.mean(inferred_batch.dict_coeff_recon.abs())
            else:
                l1_sparsity_dict_coeff = None

            l1_img_ae_loss = self.l1_img_loss(data_batch.image, inferred_batch.decoded_img)
            l2_img_ae_loss = self.l2_img_loss(data_batch.image, inferred_batch.decoded_img)
        else:
            l1_features_loss = None
            l2_features_loss = None
            l1_sparsity_features = None
            l1_sparsity_dict_coeff = None
            l1_img_ae_loss = None
            l2_img_ae_loss = None

        l1_magnitude_loss = self.l1_loss(data_batch.fft_magnitude.detach(), fft_magnitude_recon)
        l2_magnitude_loss = self.l2_loss(data_batch.fft_magnitude.detach(), fft_magnitude_recon)
        if not self._config.use_rfft:
            l2_features_realness = 0.5 * torch.mean(torch.square(inferred_batch.intermediate_features.imag.abs()))
        else:
            l2_features_realness = None

        if self._config.use_ref_net:
            l1_ref_img_recon_loss = self.l1_img_loss(data_batch.image, inferred_batch.img_recon_ref)
            l2_ref_img_recon_loss = self.l2_img_loss(data_batch.image, inferred_batch.img_recon_ref)
            l1_ref_magnitude_loss = self.l1_loss(data_batch.fft_magnitude.detach(),
                                                 inferred_batch.fft_magnitude_recon_ref)
            l2_ref_magnitude_loss = self.l2_loss(data_batch.fft_magnitude.detach(),
                                                 inferred_batch.fft_magnitude_recon_ref)
            if self._config.use_lpips:
                lpips_ref_img_recon_loss = self._lpips_loss(data_batch.image, inferred_batch.img_recon_ref)
                total_loss += self._config.lambda_ref_img_lpips * lpips_ref_img_recon_loss
            else:
                lpips_ref_img_recon_loss = None
        else:
            l2_ref_img_recon_loss = None
            l2_ref_magnitude_loss = None
            l1_ref_img_recon_loss = None
            l1_ref_magnitude_loss = None
            lpips_ref_img_recon_loss = None

        real_labels = torch.ones((data_batch.fft_magnitude.shape[0], 1),
                                 device=self.device,
                                 dtype=data_batch.fft_magnitude.dtype)

        total_loss += self._config.lambda_img_recon_loss * l2_img_recon_loss + \
                      self._config.lambda_img_recon_loss_l1 * l1_img_recon_loss

        if self._config.use_lpips:
            lpips_img_recon_loss = self._lpips_loss(data_batch.image, inferred_batch.img_recon)
            total_loss += self._config.lambda_img_lpips * lpips_img_recon_loss
        else:
            lpips_img_recon_loss = None

        l1_reg_fc_pred = None
        if len(self._config.lambda_fc_layers_pred_l1_req) > 0:
            l1_reg_fc_pred = torch.zeros_like(total_loss)
            for ind, weights_block in enumerate(self._generator_model.phase_predictor.weights_fc):
                lambda_block = self._config.lambda_fc_layers_pred_l1_req[ind] if \
                    len(self._config.lambda_fc_layers_pred_l1_req) > ind else self._config.lambda_fc_layers_pred_l1_req[-1]
                l1_reg_fc_pred += lambda_block * torch.mean(weights_block.abs())
            if self._config.lambda_fc_pred_l1_req > 0.0:
                total_loss += self._config.lambda_fc_pred_l1_req * l1_reg_fc_pred

        if use_adv_loss:
            if self._config.predict_out == 'features':
                real_img = inferred_batch.decoded_img
            else:
                real_img = data_batch.image
            gen_img_discrim_batch: DiscriminatorBatch = self.img_discriminator(inferred_batch.img_recon)
            img_adv_loss = self.adv_loss(gen_img_discrim_batch.validity, real_labels)
            total_loss += self._config.lambda_img_adv_loss * img_adv_loss

            real_img_discrim_batch: DiscriminatorBatch = self.img_discriminator(real_img)
            p_loss_discrim_img = los_fun.l2_perceptual_loss(gen_img_discrim_batch.features, real_img_discrim_batch.features,
                                                    weights=self._config.weights_plos)
            total_loss += self._config.lambda_img_perceptual_loss * p_loss_discrim_img

            if self._config.use_ref_net:
                gen_ref_img_discrim_batch: DiscriminatorBatch = self.img_discriminator(inferred_batch.img_recon_ref)
                ref_img_adv_loss = self.adv_loss(gen_ref_img_discrim_batch.validity, real_labels)
                total_loss += self._config.lambda_ref_img_adv_loss * ref_img_adv_loss

                p_loss_discrim_ref_img = los_fun.l2_perceptual_loss(gen_ref_img_discrim_batch.features,
                                                                    real_img_discrim_batch.features,
                                                                    weights=self._config.weights_plos)
                total_loss += self._config.lambda_ref_img_perceptual_loss * p_loss_discrim_ref_img
            else:
                ref_img_adv_loss = None
                p_loss_discrim_ref_img = None

            if self._config.predict_out == 'features':
                f_disc_generated_batch: DiscriminatorBatch = self.features_discriminator(inferred_batch.feature_recon_decoder)
                f_disc_real_batch: DiscriminatorBatch = self.features_discriminator(inferred_batch.feature_decoder)
                p_loss_discrim_f = los_fun.l2_perceptual_loss(f_disc_generated_batch.features,
                                                              f_disc_real_batch.features,
                                                              weights=self._config.weights_plos)
                features_adv_loss = self.adv_loss(f_disc_generated_batch.validity, real_labels)

                total_loss += self._config.lambda_features_adv_loss * features_adv_loss + \
                              self._config.lambda_features_perceptual_loss * p_loss_discrim_f
            else:
                p_loss_discrim_f = None
                features_adv_loss = None
        else:
            img_adv_loss = None
            ref_img_adv_loss = None
            features_adv_loss = None
            p_loss_discrim_f = None
            p_loss_discrim_img = None
            p_loss_discrim_ref_img = None

        total_loss += self._config.lambda_magnitude_recon_loss * l2_magnitude_loss + \
                      self._config.lambda_magnitude_recon_loss_l1 * l1_magnitude_loss

        if l2_features_realness is not None:
            total_loss += self._config.lambda_features_realness * l2_features_realness

        if self._config.predict_out == 'features':
            total_loss += self._config.lambda_features_recon_loss * l2_features_loss + \
                          self._config.lambda_features_recon_loss_l1 * l1_features_loss + \
                          self._config.lambda_sparsity_features * l1_sparsity_features
            if self._config.ae_decoder_fine_tune:
                total_loss += self._config.lambda_img_ae_loss_l2 * l2_img_ae_loss + \
                              self._config.lambda_img_ae_loss_l1 * l1_img_ae_loss

            if self._config.use_ae_dictionary:
                total_loss += self._config.lambda_sparsity_dict_coeff * l1_sparsity_dict_coeff

        if self._config.use_ref_net:
            total_loss += self._config.lambda_ref_magnitude_recon_loss_l1 * l1_ref_magnitude_loss + \
                          self._config.lambda_ref_magnitude_recon_loss * l2_ref_magnitude_loss + \
                          self._config.lambda_img_recon_loss * l2_ref_img_recon_loss + \
                          self._config.lambda_img_recon_loss_l1 * l1_ref_img_recon_loss

        losses = LossesPRFeatures(total=total_loss,
                                  l1_img=l1_img_recon_loss,
                                  l2_img=l2_img_recon_loss,
                                  l1_ae_img=l1_img_ae_loss,
                                  l2_ae_img=l2_img_ae_loss,
                                  l1_ref_img=l1_ref_img_recon_loss,
                                  l2_ref_img=l2_ref_img_recon_loss,
                                  lpips_img=lpips_img_recon_loss,
                                  lpips_ref_img=lpips_ref_img_recon_loss,
                                  l1_features=l1_features_loss,
                                  l2_features=l2_features_loss,
                                  l1_magnitude=l1_magnitude_loss,
                                  l2_magnitude=l2_magnitude_loss,
                                  l1_ref_magnitude=l1_ref_magnitude_loss,
                                  l2_ref_magnitude=l2_ref_magnitude_loss,
                                  l1_sparsity_features=l1_sparsity_features,
                                  l1_sparsity_dict_coeff=l1_sparsity_dict_coeff,
                                  realness_features=l2_features_realness,
                                  img_adv_loss=img_adv_loss,
                                  ref_img_adv_loss=ref_img_adv_loss,
                                  features_adv_loss=features_adv_loss,
                                  perceptual_disrim_features=p_loss_discrim_f,
                                  perceptual_disrim_img=p_loss_discrim_img,
                                  perceptual_disrim_ref_img=p_loss_discrim_ref_img,
                                  l1_reg_fc_pred=l1_reg_fc_pred)

        self._recon_statistics_metrics(inferred_batch, losses)

        if self._config.use_ref_net:
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
        if inferred_batch.feature_recon is not None:
            losses.mean_features = inferred_batch.feature_recon.mean()
            losses.std_features = inferred_batch.feature_recon.std()
            losses.min_features = inferred_batch.feature_recon.min()
            losses.max_features = inferred_batch.feature_recon.max()

    def _save_gan_models(self, step: int, force: bool = False) -> None:
        if self.models_path is not None:
            save_state = self._generator_model.get_state_dict()
            save_state.update({opt_name: opt_ for opt_name, opt_ in self.optimizers_generator.items()})
            if self._config.use_gan:
                save_state[ModulesNames.img_discriminator] = self.img_discriminator.state_dict()
                if self.optimizer_discr:
                    save_state[ModulesNames.opt_discriminators] = self.optimizer_discr.state_dict()
                if self._config.predict_out == 'features' and self.features_discriminator:
                    save_state[ModulesNames.features_discriminator] = self.features_discriminator.state_dict()

            if ((step - 1) % self._config.save_model_interval == 0) or force:
                save_model_path = os.path.join(self.models_path, f'phase-retrieval-gan-model-{step}.pt')
                self._log.debug(f'Save model in {save_model_path}')
                torch.save(save_state, save_model_path)

    def _log_ae_train_dbg_batch(self, step: Optional[int] = None) -> None:
        if not step:
            step = self._global_step
        with torch.no_grad():
            recon_data_tr_batch = self._generator_model.forward_ae(self.data_tr_batch)
            recon_data_ts_batch = self._generator_model.forward_ae(self.data_ts_batch)

            enc_layers_list = self._generator_model.ae_net._encoder.get_layers()
            enc_layers_features_tr = enc_layers_list(self.data_tr_batch.image, use_residual=True)[1:]
            enc_layers_features_ts = enc_layers_list(self.data_ts_batch.image, use_residual=True)[1:]

            dec_layers_list = self._generator_model.ae_net._decoder.get_layers()

            dec_layers_features_tr = dec_layers_list(self._generator_model.ae_net.bottleneck_mapping(recon_data_tr_batch.feature_encoder)[0],
                                                     use_residual=True)
            dec_layers_features_ts = dec_layers_list(self._generator_model.ae_net.bottleneck_mapping(recon_data_ts_batch.feature_encoder)[0],
                                                     use_residual=True)

        img_grid_tr = self._grid_images(self.data_tr_batch, recon_data_tr_batch)
        img_grid_ts = self._grid_images(self.data_ts_batch, recon_data_ts_batch)
        enc_layers_grid_tr = [self._build_grid_features_map(enc_layer) for enc_layer in enc_layers_features_tr]
        enc_layers_grid_ts = [self._build_grid_features_map(enc_layer) for enc_layer in enc_layers_features_ts]
        dec_layers_grid_tr = [self._build_grid_features_map(dec_layer) for dec_layer in dec_layers_features_tr]
        dec_layers_grid_ts = [self._build_grid_features_map(dec_layer) for dec_layer in dec_layers_features_ts]
        img_diff_grid_tr = self._grid_diff_images(self.data_tr_batch, recon_data_tr_batch)
        img_diff_grid_ts = self._grid_diff_images(self.data_ts_batch, recon_data_ts_batch)
        features_grid_enc_tr, features_grid_dec_tr = self._grid_features(recon_data_tr_batch)
        features_grid_enc_ts, features_grid_dec_ts = self._grid_features(recon_data_ts_batch)

        self.log_image_grid(img_grid_tr, 'train-ae/img-origin-recon', step)
        self.log_image_grid(img_grid_ts, 'test-ae/img-origin-recon', step)
        self.log_image_grid(img_diff_grid_tr, 'train-ae/img-diff-origin-recon', step)
        self.log_image_grid(img_diff_grid_ts, 'test-ae/img-diff-origin-recon', step)
        self.log_image_grid(features_grid_enc_tr, 'train-ae/features-enc', step)
        self.log_image_grid(features_grid_enc_ts, 'test-ae/features-enc', step)
        self.log_image_grid(features_grid_dec_tr, 'train-ae/features-dec', step)
        self.log_image_grid(features_grid_dec_tr, 'test-ae/features-dec', step)
        if self._config.use_ae_dictionary:
            ae_dictionary_grid = self._build_grid_features_map(self._generator_model.ae_net.get_dictionary()[None])
            self.log_image_grid(ae_dictionary_grid, 'ae_dictionary', step)

        for ind, (enc_layer_tr, enc_layer_ts) in enumerate(zip(enc_layers_grid_tr, enc_layers_grid_ts)):
            self.log_image_grid(enc_layer_tr, f'train-ae/enc_layer_{ind+1}', step)
            self.log_image_grid(enc_layer_ts, f'test-ae/enc_layer_{ind+1}', step)

        for ind, (dec_layer_tr, dec_layer_ts) in enumerate(zip(dec_layers_grid_tr, dec_layers_grid_ts)):
            self.log_image_grid(dec_layer_tr, f'train-ae/dec_layer_{ind+1}', step)
            self.log_image_grid(dec_layer_ts, f'test-ae/dec_layer_{ind+1}', step)

    def _log_en_magnitude_dbg_batch(self, use_adv_loss: bool, step: int = None) -> (LossesPRFeatures, LossesPRFeatures):
        if not step:
            step = self._global_step
        with torch.no_grad():
            inferred_batch_tr = self._generator_model.forward_magnitude_encoder(self.data_tr_batch)
            inferred_batch_ts = self._generator_model.forward_magnitude_encoder(self.data_ts_batch)

            img_grid_grid_tr, img_diff_grid_grid_tr, fft_magnitude_grid_tr, features_enc_grid_grid_tr, features_dec_grid_grid_tr = \
                self._debug_images_grids(self.data_tr_batch, inferred_batch_tr)

            img_grid_grid_ts, img_diff_grid_grid_ts, fft_magnitude_grid_ts, features_enc_grid_grid_ts, features_dec_grid_grid_ts = \
                self._debug_images_grids(self.data_ts_batch, inferred_batch_ts)

            tr_losses = self._encoder_losses(self.data_tr_batch, inferred_batch_tr, use_adv_loss=use_adv_loss)
            ts_losses = self._encoder_losses(self.data_ts_batch, inferred_batch_ts, use_adv_loss=use_adv_loss)

            self.log_image_grid(img_grid_grid_tr,
                                tag_name='train_en_magnitude/img-origin-autoencoded-recon-ref', step=step)
            self.log_image_grid(img_grid_grid_ts,
                                tag_name='test_en_magnitude/img-origin-autoencoded-recon-ref', step=step)

            self.log_image_grid(img_diff_grid_grid_tr,
                                tag_name='train_en_magnitude/img-diff-origin-autoencoded-recon-ref', step=step)
            self.log_image_grid(img_diff_grid_grid_ts,
                                tag_name='test_en_magnitude/img-diff-origin-autoencoded-recon-ref', step=step)

            self.log_image_grid(fft_magnitude_grid_tr,
                                tag_name='train_en_magnitude/fft_magnitude-origin-autoencoded-recon', step=step)
            self.log_image_grid(fft_magnitude_grid_ts,
                                tag_name='test_en_magnitude/fft_magnitude-origin-autoencoded-recon', step=step)

            self.log_image_grid(features_enc_grid_grid_tr,
                                tag_name='train_en_magnitude/features-enc-origin-recon',
                                step=step)
            self.log_image_grid(features_enc_grid_grid_ts,
                                tag_name='test_en_magnitude/features-enc-origin-recon',
                                step=step)

            self.log_image_grid(features_dec_grid_grid_tr,
                                tag_name='train_en_magnitude/features-dec-origin-recon',
                                step=step)
            self.log_image_grid(features_dec_grid_grid_ts,
                                tag_name='test_en_magnitude/features-dec-origin-recon',
                                step=step)

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
                            **kwargs) -> (str, Optional[Union[str, pd.DataFrame]]):

    config = TrainerPhaseRetrievalAeFeatures.load_config(config_path, **kwargs)

    trainer = TrainerPhaseRetrievalAeFeatures(config=config, experiment_name=experiment_name)

    trainer._log.debug(f'Generator optimizers: {trainer.optimizers_generator}')

    train_en_losses, test_en_losses, test_ae_losses = trainer.train()

    model_last_s3_path = trainer.get_last_model_s3_path()

    del trainer

    if model_last_s3_path:
        eval_test = Evaluator(model_type=model_last_s3_path).benchmark_dataset(type_ds='test')
    else:
        eval_test = None

    return model_last_s3_path, eval_test


if __name__ == '__main__':
    fire.Fire(run_ae_features_trainer)
