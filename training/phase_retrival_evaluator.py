import torch
import numpy as np
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import fire
import os
import copy
from typing import Optional, List, Callable
from tqdm import tqdm
from models import Discriminator
from torch.optim.lr_scheduler import MultiStepLR
import pandas as pd
from common import LossesPRFeatures, InferredBatch, ConfigTrainer, l2_grad_norm,  LossesGradNorms,  DiscriminatorBatch
from common import im_concatenate, l2_perceptual_loss, PATHS, DataBatch, S3FileSystem, NumpyBatch

from training.base_phase_retrieval_trainer import BaseTrainerPhaseRetrieval
from training.phase_retrieval_model import PhaseRetrievalAeModel
from training.utils import ModulesNames
from collections import defaultdict

from skimage.measure import compare_ssim as _ssim
import dataclasses
from dataclasses import dataclass, field


@dataclass
class EvaluationMetrics(NumpyBatch):
    mse: np.ndarray = field(default_factory=lambda:  np.zeros((1,)))
    mae: np.ndarray = field(default_factory=lambda:  np.zeros((1,)))
    ssim: np.ndarray = field(default_factory=lambda:  np.zeros((1,)))


class CalculateMetrics:
    @staticmethod
    def mse(true_images, predicted_images):
        return np.mean(np.square(true_images - predicted_images))

    @staticmethod
    def ssim(true_images, predicted_images):
        return _ssim(true_images.transpose(1, 2, 0), predicted_images.transpose(1, 2, 0), multichannel=True)

    @staticmethod
    def mae(true_images, predicted_images):
        return np.mean(np.abs(true_images - predicted_images))

    @staticmethod
    def metrics(true_images, predicted_images) -> EvaluationMetrics:
        mse = CalculateMetrics.mse(true_images, predicted_images)
        ssim = CalculateMetrics.ssim(true_images, predicted_images)
        mae = CalculateMetrics.mae(true_images, predicted_images)
        return EvaluationMetrics(mse=mse, ssim=ssim, mae=mae)


class TrainerPhaseRetrievalEvaluator(BaseTrainerPhaseRetrieval):
    def __init__(self, config: ConfigTrainer):
        config.use_tensor_board = False
        config.part_supervised_pairs = 1.0
        experiment_name = 'TrainerPhaseRetrievalEvaluator'
        super(TrainerPhaseRetrievalEvaluator, self).__init__(config=config, experiment_name=experiment_name)

        self._generator_model = PhaseRetrievalAeModel(config=self._config, s3=self._s3, log=self._log)
        self._generator_model.set_eval_mode()
        self._generator_model.set_device(self.device)

        loaded_sate = self.load_state()
        self._generator_model.load_modules(loaded_sate, force=True)

    def benchmark_dataset(self, test_ds: bool = True, save_out_url: Optional[str] = None):
        if test_ds:
            p_bar_data_loader = tqdm(self.test_loader)
            inv_norm = self.test_ds.get_inv_normalize_transform()
        else:
            p_bar_data_loader = tqdm(self.train_paired_loader)
            inv_norm = self.train_paired_loader.get_inv_normalize_transform()

        evaluation_metrics = []
        for batch_idx, data_batch in enumerate(p_bar_data_loader):
            data_batch = self.prepare_data_batch(data_batch)
            inferred_batch = self._generator_model.forward_magnitude_encoder(data_batch, eval_mode=False)
            gt_images = inv_norm(data_batch.image).detach().cpu().numpy()
            predicted_images = inv_norm(inferred_batch.img_recon_ref).detach().cpu().numpy()
            for gt_img , pred_img in zip(gt_images, predicted_images):
                evaluation_metrics.append(CalculateMetrics.metrics(gt_img, pred_img))

        evaluation_metrics = EvaluationMetrics.merge(evaluation_metrics)
        keys = evaluation_metrics.get_keys()
        eval_df = pd.DataFrame(index=keys, columns=['mean', 'std', 'min', 'max'])
        mean_metrics = evaluation_metrics.mean()
        std_metrics = evaluation_metrics.std()
        min_metrics = evaluation_metrics.min()
        max_metrics = evaluation_metrics.max()
        for key in evaluation_metrics.as_dict().items():
            eval_df.loc[key]['mean'] = mean_metrics.as_dict()[key]
            eval_df.loc[key]['std'] = std_metrics.as_dict()[key]
            eval_df.loc[key]['min'] = min_metrics.as_dict()[key]
            eval_df.loc[key]['max'] = max_metrics.as_dict()[key]

        self._log.debug(eval_df)
        if save_out_url is not None:
            self._s3.save_object(save_out_url, saver=lambda save_path: eval_df.to_csv(save_path))
        return eval_df






