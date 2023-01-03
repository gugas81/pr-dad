import logging
import os
from typing import Optional, Union, Tuple, Dict, Any

import fire
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

import models.losses as los_fun
from common import ConfigSpikesTrainer, DataSpikesBatch, InferredSpikesBatch, \
    LossesSpikesImages
from data.spikes_dataset import  fft_magnitude
from data import DataHolder, create_spikes_data_loaders
from models.spikes_predictors import SpikesImgReconConvModel, SpikesImgReconMlpModel
from training.base_phase_retrieval_trainer import BaseTrainerPhaseRetrieval

logging.basicConfig(level=logging.INFO)

MAX_COUNT_SPIKES = 100


class TrainerSpikesSignalRetrieval(BaseTrainerPhaseRetrieval):
    _EPS = torch.finfo(torch.float).eps

    def __init__(self, config: ConfigSpikesTrainer, experiment_name: str):
        super(TrainerSpikesSignalRetrieval, self).__init__(config=config, experiment_name=experiment_name)
        self._config: ConfigSpikesTrainer = self._config

        self._l2_loss = nn.MSELoss()
        self._l1_loss = nn.L1Loss()

        self._loss_recon_img_fun = los_fun.LossImg(loss_type=self._config.loss_type_img_recon,
                                                   rot180=True,
                                                   device=self.device)
        self._loss_recon_fft_fun = los_fun.LossImg(loss_type=config.loss_type_mag,
                                                   rot180=False,
                                                   device=self.device)

        self._sparsity_loss = los_fun.SparsityL1Loss()
        self._support_size_fun = lambda x: x.abs().sum((-1, -2, -3)).unsqueeze(1)  # los_fun.SparsityL1Loss(reduction='sum')
        self._support_loss_fun = nn.L1Loss()

        self._model: nn.Module = self._create_predictor_model()
        self.load_models()

        self._model.to(device=self.device)
        self._model.train()

        self._optimizer = optim.Adam(params=self._model.parameters(), lr=self._config.learning_rate)
        self._lr_schedulers = MultiStepLR(self._optimizer, self._config.lr_milestones, 0.5)

    def _init_dbg_data_batches(self):
        self.data_ts_batch = self.get_batch_test().get_subset(self._config.dbg_img_batch)

    def get_batch_test(self) -> DataSpikesBatch:
        batch_data = next(iter(self._data_holder.test_loader))
        batch_data: DataSpikesBatch = DataSpikesBatch.from_dict(batch_data).to(self.device)
        return batch_data

    def get_model(self) -> nn.Module:
        return self._model

    def load_models(self):
        if self._config.path_pretrained is not None:
            loaded_sate = self.load_state(self._config.path_pretrained)
            self._model.load_state_dict(loaded_sate)
            self._log.debug(f'Model parameters was loaded from file: {self._config.path_pretrained}')

    def _create_predictor_model(self) -> nn.Module:
        if self._config.model_type == 'mlp':
            model = SpikesImgReconMlpModel(img_size=self._config.image_size,
                                           spikes_meta_size=1,
                                           tile_size=self._config.tile_size,
                                           is_proj_mag=self._config.proj_mag,
                                           fft_shifted=self._config.shift_fft)

        elif self._config.model_type == 'conv_unet' or self._config.model_type == 'conv_ae':
            model = SpikesImgReconConvModel(img_size=self._config.image_size,
                                            spikes_meta_size=1,
                                            tile_size=self._config.tile_size,
                                            is_proj_mag=self._config.proj_mag,
                                            pred_type=self._config.model_type,
                                            count_predictor=self._config.count_predictor_head)
        else:
            raise self._log.error(f'Non valid model type: {self._config.model_type}')
        return model

    def _init_data_loaders(self) -> DataHolder:
        return create_spikes_data_loaders(config=self._config,
                                          s3=self._s3,
                                          log=self._log,
                                          inv_norm=TrainerSpikesSignalRetrieval.img_norm_min_max)

    def prepare_data_batch(self, data_batch: Dict[str, Any], is_train: bool = True) -> DataSpikesBatch:
        data_batch: DataSpikesBatch = DataSpikesBatch.from_dict(data_batch).to(device=self.device)
        return data_batch

    def prepare_dbg_batch(self,
                          data_batch: Tensor,
                          num_images: Optional[int] = None,
                          image_size: Optional[int] = None,
                          grid_ch: bool = False) -> np.ndarray:

        raise NotImplemented

    def forward(self, batch_data: DataSpikesBatch) -> InferredSpikesBatch:
        n_spikes_emb = batch_data.n_spikes / MAX_COUNT_SPIKES
        input_data = batch_data.fft_magnitude_noised if self._config.use_noised_input else batch_data.fft_magnitude
        img_spikes_pred, pred_n_spikes = self._model(input_data, n_spikes_emb)
        fft_spikes_pred = fft_magnitude(img_spikes_pred, shift=self._config.shift_fft)
        inferred_batch = InferredSpikesBatch(img_recon=img_spikes_pred,
                                             fft_recon=fft_spikes_pred,
                                             pred_n_spikes=pred_n_spikes)
        return inferred_batch

    def calc_losses(self, data_batch: DataSpikesBatch, inferred_batch: InferredSpikesBatch) -> LossesSpikesImages:
        img_spikes_norm = self.img_norm(data_batch.image)
        img_spikes_pred_norm = self.img_norm(inferred_batch.img_recon)

        fft_spikes_norm = self.fft_norm(data_batch.fft_magnitude)
        fft_spikes_pred_norm = self.fft_norm(inferred_batch.fft_recon)

        fft_recon_loss = self._loss_recon_fft_fun(fft_spikes_norm, fft_spikes_pred_norm)

        img_recon_support_size = self._support_size_fun(inferred_batch.img_recon)
        support_size_loss = self._support_loss_fun(data_batch.n_spikes.to(dtype=img_recon_support_size.dtype),
                                                   img_recon_support_size)

        img_recon_loss = self._loss_recon_img_fun(img_spikes_norm, img_spikes_pred_norm)
        img_recon_sparsity_loss = self._sparsity_loss(inferred_batch.img_recon)

        total_loss = self._config.lambda_img_recon * img_recon_loss + \
                     self._config.lambda_sparsity * img_recon_sparsity_loss + \
                     self._config.lambda_fft_recon * fft_recon_loss + \
                     self._config.lambda_support_size * support_size_loss

        if self._config.count_predictor_head:
            n_spikes_emb = data_batch.n_spikes / MAX_COUNT_SPIKES
            count_pred_loss = self._l1_loss(n_spikes_emb, inferred_batch.pred_n_spikes)
            total_loss += self._config.lambda_count_spikes * count_pred_loss
        else:
            count_pred_loss = None

        losses = LossesSpikesImages(total=total_loss,
                                    img_recon=img_recon_loss,
                                    img_sparsity=img_recon_sparsity_loss,
                                    fft_recon=fft_recon_loss,
                                    count_pred_loss=count_pred_loss,
                                    support_size=support_size_loss)

        return losses

    def eval_model(self) -> LossesSpikesImages:
        losses_eval = []
        for batch_eval in tqdm(self._data_holder.test_loader):
            batch_data_eval: DataSpikesBatch = DataSpikesBatch.from_dict(batch_eval)
            batch_data_eval = batch_data_eval.to(self.device)
            inferred_batch_eval = self.forward(batch_data_eval)
            losses_eval_ = self.calc_losses(batch_data_eval, inferred_batch_eval)
            losses_eval.append(losses_eval_.detach())

        losses_eval = LossesSpikesImages.merge(losses_eval)
        self._data_holder.test_loader
        return losses_eval

    def images_eval(self) -> (InferredSpikesBatch, LossesSpikesImages, DataSpikesBatch, InferredSpikesBatch):
        batch_inferred_test = self.forward(self.data_ts_batch)
        batch_data_rand = self.get_batch_test().get_subset(self._config.dbg_img_batch)
        batch_inferred_rand = self.forward(batch_data_rand)
        losses_eval_test = self.calc_losses(self.data_ts_batch, batch_inferred_test)
        return batch_inferred_test, losses_eval_test, batch_data_rand, batch_inferred_rand

    def train_model(self):
        self._global_step = 0
        for epoch in range(self._config.n_epochs_pr):
            losses_tr = []
            self._model.train()
            self._log.info(f'====Train epoch: {epoch}====')
            len_train_loader = len(self._data_holder.train_paired_loader) * self._config.batch_size_train
            for ind, batch_data in tqdm(enumerate(self._data_holder.train_paired_loader)):
                batch_data: DataSpikesBatch = DataSpikesBatch.from_dict(batch_data)

                losses_tr_step = self.train_step(batch_data)
                losses_tr.append(losses_tr_step.detach())

                if ind % self._config.log_interval == 0:
                    self._log.info(f' epoch: {epoch}, step: {ind}, {100 * ind/len_train_loader: .0f}%, '
                                   f'step_tr_losses: {losses_tr_step}')
                    self._add_losses_tensorboard('spikes-pred/train_step', losses_tr_step, self._global_step)

                if ind % self._config.log_image_interval == 0:
                    self._model.eval()
                    self._log_train_dbg_batch(step=self._global_step)
                    self._model.train()

                self._global_step += 1
            losses_tr: LossesSpikesImages = LossesSpikesImages.merge(losses_tr)
            curr_lr = self._lr_schedulers.get_last_lr()
            losses_tr.lr = torch.tensor(curr_lr[0])
            losses_tr = losses_tr.detach()
            self._log.info(f'Eval epoch: {epoch}, step_tr_losses: {losses_tr}')
            self._add_losses_tensorboard('spikes-pred/epoch_tr', losses_tr, self._global_step)

            self._log.info(f'Run eval epoch: {epoch}')
            self._model.eval()
            losses_eval = self.eval_model()
            self._log.info(f'Epoch: {epoch}, eval_losses: {losses_eval}')
            self._add_losses_tensorboard('spikes-pred/eval', losses_eval, self._global_step)
            self._model.train()

            if (epoch % self._config.save_model_interval == 0) or (epoch == self._config.n_epochs_pr - 1):
                save_model_path = os.path.join(self.models_path, f'phase-retrieval-spikes-pred-model-{epoch}.pt')
                self._log.debug(f'Save model in {save_model_path}')
                torch.save(self._model.state_dict(), save_model_path)

            self._lr_schedulers.step()

    def _log_train_dbg_batch(self, step: int) -> None:
        batch_inferred_test, losses_eval_test, batch_data_rand, batch_inferred_rand = self.images_eval()
        img_grid_grid_ts, img_diff_grid_grid_ts, fft_magnitude_grid_ts, _, _ = \
            self._debug_images_grids(self.data_ts_batch, batch_inferred_test,
                                     norm_fun=TrainerSpikesSignalRetrieval.img_norm_min_max,
                                     normalize_img=False, normalize_fft=False)
        img_grid_grid_rnd, img_diff_grid_grid_rnd, fft_magnitude_grid_rnd, _, _ = \
            self._debug_images_grids(batch_data_rand, batch_inferred_rand,
                                     norm_fun=TrainerSpikesSignalRetrieval.img_norm_min_max,
                                     normalize_img=False, normalize_fft=False)

        self.log_image_grid(img_grid_grid_ts,
                            tag_name='train_spikes-pred_ts/img-origin-noised-recon', step=step)
        self.log_image_grid(fft_magnitude_grid_ts,
                            tag_name='train_spikes-pred_ts/fft-origin-noised-recon', step=step)
        self.log_image_grid(img_grid_grid_rnd,
                            tag_name='train_spikes-pred_rnd/img-origin-noised-recon', step=step)
        self.log_image_grid(fft_magnitude_grid_rnd,
                            tag_name='train_spikes-pred_rnd/fft-origin-noised-recon', step=step)

        self._log.info(f'step: {step}, losses_eval_test: {losses_eval_test}')
        self._add_losses_tensorboard('spikes-pred/test_step', losses_eval_test, step)

    def train_step(self, batch_data: DataSpikesBatch) -> LossesSpikesImages:
        self._optimizer.zero_grad()
        batch_data = batch_data.to(self.device)
        inferred_batch = self.forward(batch_data)
        losses_tr_step = self.calc_losses(batch_data, inferred_batch)
        losses_tr_step.total.backward()
        self._optimizer.step()
        return losses_tr_step

    @staticmethod
    def img_norm(img_: Tensor) -> Tensor:
        img_norm_ = img_ - img_.mean((-2, -1), keepdim=True)
        img_norm_ = img_norm_ / (img_norm_.std((-2, -1), keepdim=True) + TrainerSpikesSignalRetrieval._EPS)
        return img_norm_

    @staticmethod
    def img_norm_min_max(img_: Tensor) -> Tensor:
        img_norm_ = img_ - img_.min()
        img_norm_ = img_norm_ / (img_norm_.max() + TrainerSpikesSignalRetrieval._EPS)
        return img_norm_

    @staticmethod
    def fft_norm(mag: Tensor) -> Tensor:
        mag_norm = mag / (mag.mean((-2, -1), keepdim=True) + TrainerSpikesSignalRetrieval._EPS)
        return mag_norm


def run_pr_spikes_trainer(experiment_name: str = 'recon-spikes-signals',
                          config_path: str = None,
                          **kwargs):
    config = ConfigSpikesTrainer.load_config(config_path, **kwargs)
    trainer = TrainerSpikesSignalRetrieval(config=config, experiment_name=experiment_name)
    log = trainer.get_log()
    log.debug(f'Train spikes model: {trainer.get_model()}')
    trainer.train_model()


if __name__ == '__main__':
    fire.Fire()
