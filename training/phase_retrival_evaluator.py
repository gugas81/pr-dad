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

from skimage.metrics import structural_similarity as _ssim
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
    def __init__(self, model_path: str, config_path: str):
        config = self.load_config(config_path)
        config.use_tensor_board = False
        config.part_supervised_pairs = 1.0
        config.batch_size_test = 128
        config.load_modules = ['all']
        experiment_name = 'TrainerPhaseRetrievalEvaluator'
        super(TrainerPhaseRetrievalEvaluator, self).__init__(config=config, experiment_name=experiment_name)
        self._generator_model = PhaseRetrievalAeModel(config=self._config, s3=self._s3, log=self._log)
        self._generator_model.set_eval_mode()
        self._generator_model.set_device(self.device)
        loaded_sate = self.load_state(model_path)
        self._generator_model.load_modules(loaded_sate, force=True)

    def benchmark_dataset(self, save_out_url: str, test_ds: bool = True):
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
        for key in keys:
            eval_df.loc[key]['mean'] = mean_metrics.as_dict()[key]
            eval_df.loc[key]['std'] = std_metrics.as_dict()[key]
            eval_df.loc[key]['min'] = min_metrics.as_dict()[key]
            eval_df.loc[key]['max'] = max_metrics.as_dict()[key]

        self._log.debug(f'\n{eval_df}')
        if save_out_url is not None:
            save_out_url_csv = os.path(save_out_url, 'eval-metrics.csv')
            self._s3.save_object(save_out_url_csv, saver=lambda save_path: eval_df.to_csv(save_path))
            self.eval_dbg_batch(save_out_url)
            return save_out_url
        else:
            return eval_df

    def eval_dbg_batch(self, save_url: str):
        inferred_batch_tr = self._generator_model.forward_magnitude_encoder(self.data_tr_batch, eval_mode=False)
        inferred_batch_ts = self._generator_model.forward_magnitude_encoder(self.data_ts_batch, eval_mode=False)

        img_grid_grid_tr, img_diff_grid_grid_tr, fft_magnitude_grid_tr, features_grid_grid_tr = \
            self._debug_images_grids(self.data_tr_batch, inferred_batch_tr)

        img_grid_grid_ts, img_diff_grid_grid_ts, fft_magnitude_grid_ts, features_grid_grid_ts = \
            self._debug_images_grids(self.data_ts_batch, inferred_batch_ts)

        self._save_img_to_s3(img_grid_grid_tr, os.path.join(save_url, f'tr-img-origin-ae-recon-ref.png'))
        self._save_img_to_s3(img_grid_grid_ts, os.path.join(save_url, f'ts-img-origin-ae-recon-ref.png'))

        self._save_img_to_s3(fft_magnitude_grid_tr,
                             os.path.join(save_url, f'tr-fft_magnitude-origin-ae-recon.png'))
        self._save_img_to_s3(fft_magnitude_grid_ts,
                             os.path.join(save_url, f'ts-fft_magnitude-origin-ae-recon.png'))

        self._save_img_to_s3(img_diff_grid_grid_tr,
                             os.path.join(save_url, f'tr-img-diff-origin-ae-recon-ref.png'))
        self._save_img_to_s3(img_diff_grid_grid_ts,
                             os.path.join(save_url, f'ts-img-diff-origin-ae-recon-ref.png'))

        self._save_img_to_s3(features_grid_grid_tr,
                             os.path.join(save_url, f'tr-features-origin-recon.png'))
        self._save_img_to_s3(features_grid_grid_ts,
                             os.path.join(save_url, f'ts-features-origin-recon.png'))


if __name__ == '__main__':
    fire.Fire(TrainerPhaseRetrievalEvaluator)




