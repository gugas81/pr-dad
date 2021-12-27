import numpy as np
import fire
import math
import os
import torch
from tqdm import tqdm
from typing import List, Optional, Callable, Union
import pandas as pd
from dataclasses import dataclass, field
from skimage.metrics import structural_similarity as _ssim
from copy import deepcopy

from common import DataBatch, InferredBatch, NumpyBatch

from training.base_phase_retrieval_trainer import BaseTrainerPhaseRetrieval, DataHolder
from training.phase_retrieval_model import PhaseRetrievalAeModel


@dataclass
class EvaluationMetrics(NumpyBatch):
    mse: np.ndarray = field(default_factory=lambda:  np.zeros((1,)))
    mae: np.ndarray = field(default_factory=lambda:  np.zeros((1,)))
    ssim: np.ndarray = field(default_factory=lambda:  np.zeros((1,)))
    psnr: np.ndarray = field(default_factory=lambda:  np.zeros((1,)))


class CalculateMetrics:
    @staticmethod
    def _err_loss(gt_img: np.ndarray, predicted_img: np.ndarray, norm_fun: Callable, rot180: bool = False):
        if rot180:
            gt_images_180_rot = np.rot90(gt_img, 2, (-2, -1))
            gt_images_2d = np.stack((gt_images_180_rot, gt_img), -1)
            predicted_img_2 = np.expand_dims(predicted_img, -1)
            loss = np.mean(norm_fun(gt_images_2d - predicted_img_2), (-3, -2))
            loss = np.min(loss, -1)
            loss = loss.mean()
        else:
            loss = np.mean(norm_fun(gt_img - predicted_img))
        return loss

    @staticmethod
    def mse(gt_img: np.ndarray, predicted_img: np.ndarray, rot180: bool = False) -> np.ndarray:
        loss = CalculateMetrics._err_loss(gt_img, predicted_img, norm_fun=np.square, rot180=rot180)
        return loss

    @staticmethod
    def ssim(gt_img: np.ndarray, predicted_img: np.ndarray, rot180: bool = False) -> np.ndarray:
        predicted_img_hwc = predicted_img.transpose(1, 2, 0)
        ssim_loss = _ssim(gt_img.transpose(1, 2, 0), predicted_img_hwc, multichannel=True)
        if rot180:
            gt_images_180_rot_hwc = np.rot90(gt_img, 2, (-2, -1)).transpose(1, 2, 0)
            ssim_loss_180_rot = _ssim(gt_images_180_rot_hwc, predicted_img_hwc, multichannel=True)
            ssim_loss = np.max(np.stack((ssim_loss, ssim_loss_180_rot), axis=-1), axis=-1)
        return ssim_loss

    @staticmethod
    def mae(gt_img: np.ndarray, predicted_img: np.ndarray, rot180: bool = False) -> np.ndarray:
        loss = CalculateMetrics._err_loss(gt_img, predicted_img, norm_fun=np.abs, rot180=rot180)
        return loss

    @staticmethod
    def metrics(gt_img: np.ndarray, predicted_img: np.ndarray, rot180: bool = False) -> EvaluationMetrics:
        mse = CalculateMetrics.mse(gt_img, predicted_img, rot180=rot180)
        ssim = CalculateMetrics.ssim(gt_img, predicted_img, rot180=rot180)
        mae = CalculateMetrics.mae(gt_img, predicted_img, rot180=rot180)
        return EvaluationMetrics(mse=mse, ssim=ssim, mae=mae)


class Evaluator(BaseTrainerPhaseRetrieval):
    EVAL_MODE = True
    MEAN = 'mean'
    STD = 'std'
    MIN = 'min'
    MAX = 'max'

    COLUMNS = [MEAN, STD, MIN, MAX]

    RECON = 'recon'
    RECON_REF = 'recon_ref_net'
    AE = 'ae'
    MAIN_KEYS = [RECON_REF, RECON, AE]

    def __init__(self,
                 model_type:  Union[str, PhaseRetrievalAeModel],
                 config: Optional[str] = None,
                 data_holder: Optional[DataHolder] = None,
                 debug: bool = False,
                 rot180: bool = False):

        if isinstance(model_type, str):
            self._log.info(f'load state of the model from:  {model_type}')
            loaded_sate = self.load_state(model_type)
            assert ('config' in loaded_sate) or isinstance(config, str)
        else:
            assert isinstance(model_type, PhaseRetrievalAeModel)
            loaded_sate = None

        if isinstance(model_type, PhaseRetrievalAeModel):
            config = deepcopy(model_type._config)
        else:
            config_obj = loaded_sate['config'] if 'config' in loaded_sate else config
            config = self.load_config(config_obj)

        config.use_tensor_board = False
        config.use_gan = False
        config.part_supervised_pairs = 1.0
        config.batch_size_test = 128
        config.use_aug = False
        config.use_aug_test = False
        config.load_modules = ['all']
        config.loss_rot180 = (config.loss_rot180 or rot180)
        if debug:
            config.n_dataloader_workers = 0

        super(Evaluator, self).__init__(config=config, data_holder=data_holder)

        if isinstance(model_type, PhaseRetrievalAeModel):
            self._generator_model = model_type
        else:
            self._log.info(f'Build generator_model for config: \n  {self._config} \n')
            self._generator_model = PhaseRetrievalAeModel(config=self._config, s3=self._s3, log=self._log)

        if self.EVAL_MODE:
            self._generator_model.set_eval_mode()
        else:
            self._generator_model.set_train_mode()

        self._generator_model.set_device(self.device)

        self._log.debug(f'loaded_sate from {model_type}')

        if loaded_sate:
            self._generator_model.load_modules(loaded_sate, force=True)

    def benchmark_evaluation(self, save_out_url: str):
        if self._s3.exists(save_out_url):
            self._log.warning(f'Exists save_out_url: {save_out_url}')

        self._log.debug(f'Benchmark evaluation will be saved in {save_out_url}')
        self.eval_dbg_batch(save_out_url)

        self._log.debug('benchmark test dataset')
        df_ts_url_csv = self.benchmark_dataset(save_out_url, type_ds='test')
        self._log.debug(f'Eval table was saved in {df_ts_url_csv}')

        self._log.debug('benchmark train dataset')
        df_ts_url_csv = self.benchmark_dataset(save_out_url, type_ds='train')
        self._log.debug(f'Eval table was saved in {df_ts_url_csv}')

    def benchmark_dataset(self, save_out_url: Optional[str] = None, type_ds: str = 'test') -> Union[str, pd.DataFrame]:
        if type_ds == 'test':
            data_loader = self._data_holder.test_loader
            inv_norm = self._data_holder.test_ds.get_inv_normalize_transform()
        elif type_ds == 'train':
            data_loader = self._data_holder.train_paired_loader
            inv_norm = self._data_holder.train_ds.get_inv_normalize_transform()
        else:
            raise NameError(f'Non valid ds type: {type_ds}, must train/test')

        p_bar_data_loader = tqdm(data_loader)
        eval_metrics_recon_ref_net = []
        eval_metrics_recon = []
        eval_metrics_ae = []
        rot_180 = self._config.loss_rot180
        max_gt_value = float('-inf')
        for batch_idx, data_batch in enumerate(p_bar_data_loader):
            with torch.no_grad():
                data_batch = self.prepare_data_batch(data_batch)
                inferred_batch = self._generator_model.forward_magnitude_encoder(data_batch, eval_mode=self.EVAL_MODE)
                gt_images = inv_norm(data_batch.image).detach().cpu().numpy()
                max_gt_value = max(max_gt_value, float(np.max(gt_images)))
                recon_ref_images = inv_norm(inferred_batch.img_recon_ref).detach().cpu().numpy()
                recon_images = inv_norm(inferred_batch.img_recon).detach().cpu().numpy()
            if self._config.predict_out == 'features':
                ae_images = inv_norm(inferred_batch.decoded_img).detach().cpu().numpy()
            else:
                ae_images = np.zeros_like(gt_images)
            for gt_img, recon_ref_img, recon_img, ae_img in zip(gt_images, recon_ref_images, recon_images, ae_images):
                eval_metrics_recon_ref_net.append(CalculateMetrics.metrics(gt_img, recon_ref_img, rot_180))
                eval_metrics_recon.append(CalculateMetrics.metrics(gt_img, recon_img, rot_180))
                if self._config.predict_out == 'features':
                    eval_metrics_ae.append(CalculateMetrics.metrics(gt_img, ae_img))

        eval_recon_ref_net_df = self._get_eval_df(eval_metrics_recon_ref_net, max_val=max_gt_value)
        eval_recon_df = self._get_eval_df(eval_metrics_recon, max_val=max_gt_value)
        eval_df = [eval_recon_ref_net_df, eval_recon_df]
        keys = ['recon_ref_net', 'recon']
        if self._config.predict_out == 'features':
            eval_ae_df = self._get_eval_df(eval_metrics_ae, max_val=max_gt_value)
            eval_df.append(eval_ae_df)
            keys.append('ae')

        eval_df = pd.concat(eval_df, keys=keys)

        self._log.debug(f'\n{eval_df}')

        if save_out_url:
            save_out_url_csv = os.path.join(save_out_url, f'eval-metrics-{type_ds}.csv')
            self._s3.save_object(save_out_url_csv, saver=lambda save_path: eval_df.to_csv(save_path))
            self.eval_dbg_batch(save_out_url)
            out_eval = save_out_url_csv
        else:
            out_eval = eval_df
        return out_eval

    @staticmethod
    def _get_eval_df(evaluation_metrics: List[EvaluationMetrics], max_val: float) -> pd.DataFrame:
        evaluation_metrics = EvaluationMetrics.merge(evaluation_metrics)
        keys = evaluation_metrics.get_keys()
        eval_df = pd.DataFrame(index=keys, columns=Evaluator.COLUMNS)
        mean_metrics = evaluation_metrics.mean()
        std_metrics = evaluation_metrics.std()
        min_metrics = evaluation_metrics.min()
        max_metrics = evaluation_metrics.max()
        for key in keys:
            eval_df.loc[key][Evaluator.MEAN] = mean_metrics.as_dict()[key]
            eval_df.loc[key][Evaluator.STD] = std_metrics.as_dict()[key]
            eval_df.loc[key][Evaluator.MIN] = min_metrics.as_dict()[key]
            eval_df.loc[key][Evaluator.MAX] = max_metrics.as_dict()[key]

        eval_df.loc['psnr'][Evaluator.MEAN] = Evaluator.psnr(eval_df.loc['mse'][Evaluator.MEAN])

        return eval_df

    @staticmethod
    def psnr(mse: float, max_val: float) -> float:
        return 20.0 * math.log10(max_val / mse)

    def eval_dbg_batch(self, save_url: str):
        inferred_batch_tr = self._generator_model.forward_magnitude_encoder(self.data_tr_batch,
                                                                            eval_mode=self.EVAL_MODE)
        inferred_batch_ts = self._generator_model.forward_magnitude_encoder(self.data_ts_batch,
                                                                            eval_mode=self.EVAL_MODE)
        self._save_dbg_img(self.data_tr_batch, inferred_batch_tr, save_url, 'tr')
        self._save_dbg_img(self.data_ts_batch, inferred_batch_ts, save_url, 'ts')

    def _save_dbg_img(self, data_batch: DataBatch, inferred_batch: InferredBatch, save_url: str, prefix: str):
        img_grid_grid, img_diff_grid_grid, fft_magnitude_grid, features_enc_grid_grid, features_dec_grid_grid = \
            self._debug_images_grids(data_batch, inferred_batch, normalize_img=False)
        self._save_img_to_s3(img_grid_grid,
                             os.path.join(save_url, f'{prefix}-img-origin-ae-recon-ref.png'))
        self._save_img_to_s3(fft_magnitude_grid,
                             os.path.join(save_url, f'{prefix}-fft_magnitude-origin-ae-recon.png'))
        self._save_img_to_s3(img_diff_grid_grid,
                             os.path.join(save_url, f'{prefix}-img-diff-origin-ae-recon-ref.png'))
        if features_enc_grid_grid is not None:
            self._save_img_to_s3(features_enc_grid_grid,
                                 os.path.join(save_url, f'{prefix}-features-enc-origin-recon.png'))
        if features_dec_grid_grid is not None:
            self._save_img_to_s3(features_dec_grid_grid,
                                 os.path.join(save_url, f'{prefix}-features-dec-origin-recon.png'))


if __name__ == '__main__':
    fire.Fire(Evaluator)




