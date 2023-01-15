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
from torch.utils.data import DataLoader
from tqdm import tqdm

import models.losses as los_fun
from common import im_concatenate, im_save, S3FileSystem
from data.spikes_dataset import SpikesDataGenerator, fft_magnitude
from models.spikes_predictors import SpikesCountPredictor, SpikesImgReconConvModel, SpikesImgReconMlpModel

logging.basicConfig(level=logging.INFO)

MAX_COUNT_SPIKES = 100

def norm_img(img_arr: np.ndarray) -> np.ndarray:
    img_arr -= img_arr.min()
    img_arr /= max(img_arr.max(), np.finfo(img_arr.dtype).eps)
    return img_arr

def norm_mag(fft_spikes_mag: Tensor) -> Tensor:
    return fft_spikes_mag / (fft_spikes_mag.mean((-2, -1), keepdim=True) + torch.finfo(fft_spikes_mag.dtype).eps)



def run_pr_spikes_img(img_size: int = 32,
                      tile_size: int = 16,
                      n_iters_tr: int = 10000,
                      out_save_dir: Optional[str] = None,
                      pred_type: str = 'mlp',
                      spikes_range: Union[int, Tuple[int, int]] = 5,
                      epochs: int = 20,
                      noised_mag: bool = True,
                      is_proj_mag: bool = False,
                      shift_fft: bool = False,
                      dbg_mode: bool = False):
    def to_numpy_img(batch_tensor: Tensor, batch_size_: int) -> np.ndarray:
        batch_size_curr = min(batch_size_, batch_tensor.shape[0])
        img_np = norm_img(batch_tensor.detach().cpu().numpy())[:batch_size_curr, ..., None]
        return im_concatenate(img_np)

    def img_norm(img_):
        img_norm_ = img_ - img_.mean((-2, -1), keepdim=True)
        img_norm_ = img_norm_ / (img_norm_.std((-2, -1), keepdim=True) + eps)
        return img_norm_

    log = logging.getLogger('PR-IMAGES-TRAINER')
    s3 = S3FileSystem()

    n_iters_eval = 100
    batch_size = 16
    batch_size_eval = 4
    num_workers = 8
    if dbg_mode:
        num_workers = 0
        batch_size = 4

    lr = 0.0001
    add_gauss_noise = 0.005
    sigma = 0.75
    loss_type_img_recon = 'l1'
    loss_type_mag = 'l2'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eps = torch.finfo(torch.float).eps

    # spikes_regressor: MlpNet = run_train_count_spikes(spikes_range=spikes_range, batch_size=batch_size, img_size=img_size)
    #
    # spikes_embd = lambda x: spikes_regressor.fc_layers[:-2](x)

    spike_generator = SpikesDataGenerator(spikes_range=spikes_range,
                                          img_size=img_size,
                                          add_gauss_noise=add_gauss_noise,
                                          sigma=sigma,
                                          len_ds=n_iters_tr * batch_size,
                                          shift_fft=shift_fft)

    spikes_loader = DataLoader(spike_generator, batch_size=batch_size, num_workers=num_workers)
    spikes_loader_val = DataLoader(spike_generator, batch_size=batch_size, num_workers=num_workers)

    loss_recon_img_fun = los_fun.LossImg(loss_type=loss_type_img_recon, rot180=True, device=device)
    loss_recon_fft_fun = los_fun.LossImg(loss_type=loss_type_mag, rot180=False, device=device)
    sigmoid = nn.Identity()  # nn.Sigmoid()
    lambda_support_size = 1.0  # 0.01
    lambda_img_recon = 20.0  # 2.0

    lambda_fft_recon = 2.0  # 20.0
    lambda_sparsity = 0.1

    if is_proj_mag:
        lambda_fft_recon = 0.0

    sparsity_f_loss = los_fun.SparsityL1Loss()
    support_size_fun = lambda x: x.abs().sum((1, 2))[:, None]  # los_fun.SparsityL1Loss(reduction='sum')
    support_loss_fun = torch.nn.L1Loss()

    log.info(f'device: {device}, img_size={img_size}, spikes_range={spikes_range}, n_iters={n_iters_tr},'
             f'proj_mag: {is_proj_mag}, lr: {lr}')

    if pred_type == 'mlp':
        spikes_img_predictor = SpikesImgReconMlpModel(img_size=img_size,
                                                      spikes_meta_size=1,
                                                      tile_size=tile_size,
                                                      is_proj_mag=is_proj_mag,
                                                      fft_shifted=shift_fft).to(device=device)
    elif pred_type == 'conv_unet' or pred_type == 'conv_ae':
        spikes_img_predictor = SpikesImgReconConvModel(img_size=img_size,
                                                       spikes_meta_size=1,
                                                       tile_size=tile_size,
                                                       is_proj_mag=is_proj_mag,
                                                       pred_type=pred_type).to(device=device)
    else:
        raise NameError(f'Non valid pred_type: {pred_type}')

    log.info(f'{spikes_img_predictor}')

    pred_optimizer = optim.Adam(params=spikes_img_predictor.parameters(), lr=lr)
    lr_milestones = [15, 20, 25, 30, 32]

    lr_scheduler = MultiStepLR(pred_optimizer, lr_milestones, 0.5)

    def eval_model():
        spikes_img_predictor.eval()
        img_recon_loss_eval = []
        img_recon_sparsity_eval = []
        fft_recon_loss_eval = []
        support_loss_eval = []
        total_loss_eval = []
        for _, batch_eval in zip(range(n_iters_eval), spikes_loader_val):
            fft_spikes = batch_eval['fft_spikes'].to(dtype=torch.float, device=device)
            fft_spikes_noised = batch_eval['fft_spikes_noised'].to(dtype=torch.float, device=device)
            # fft_spikes_noised_flatten = torch.flatten(fft_spikes_noised, 1)
            img_spikes = batch_eval['img_spikes'].to(dtype=torch.float, device=device)
            n_spikes = batch_eval['n_spikes'].to(dtype=torch.float, device=device)
            n_spikes_emb = n_spikes / MAX_COUNT_SPIKES
            img_spikes_noised_norm = img_norm(img_spikes_noised)
            img_spikes_norm = img_norm(img_spikes)

            magnitude = fft_spikes_noised if noised_mag else fft_spikes
            # if not shift_fft:
            #     magnitude = torch.fft.fftshift(magnitude)
            img_spikes_pred = spikes_img_predictor(magnitude, n_spikes_emb)
            img_spikes_pred_norm = img_norm(img_spikes_pred)

            img_recon_support_size = support_size_fun(img_spikes_pred)
            support_loss = support_loss_fun(n_spikes, img_recon_support_size)

            fft_spikes_pred = fft_magnitude(img_spikes_pred, shift=shift_fft)
            fft_spikes_norm = fft_spikes / (fft_spikes.mean((-2, -1), keepdim=True) + eps)
            fft_spikes_pred_norm = fft_spikes_pred / (fft_spikes_pred.mean((-2, -1), keepdim=True) + eps)
            loss_fft_val = loss_recon_fft_fun(fft_spikes_norm, fft_spikes_pred_norm)

            img_recon_loss_eval.append(loss_recon_img_fun(img_spikes_norm, img_spikes_pred_norm).detach().cpu().numpy())
            img_recon_sparsity_eval.append(sparsity_f_loss(sigmoid(img_spikes_pred)).detach().cpu().numpy())
            fft_recon_loss_eval.append(loss_fft_val.detach().cpu().numpy())
            support_loss_eval.append(support_loss.detach().cpu().numpy())
            total_loss_eval.append(
                lambda_img_recon * img_recon_loss_eval[-1] + lambda_sparsity * img_recon_sparsity_eval[-1] + \
                lambda_fft_recon * fft_recon_loss_eval[-1] + lambda_support_size * support_loss_eval[-1])

        total_loss_eval = np.mean(np.array(total_loss_eval))
        fft_recon_loss_eval = np.mean(np.array(fft_recon_loss_eval))
        img_recon_sparsity_eval = np.mean(np.array(img_recon_sparsity_eval))
        img_recon_loss_eval = np.mean(np.array(img_recon_loss_eval))
        support_loss_eval = np.mean(np.array(support_loss_eval))

        return total_loss_eval, img_recon_loss_eval, img_recon_sparsity_eval, fft_recon_loss_eval, support_loss_eval

    batch_data_test = next(iter(spikes_loader_val))

    for epoch in range(epochs):
        for ind, batch_spikes_data in zip(range(n_iters_tr), spikes_loader):
            pred_optimizer.zero_grad()
            fft_spikes = batch_spikes_data['fft_spikes'].to(dtype=torch.float, device=device)
            fft_spikes_noised = batch_spikes_data['fft_spikes_noised'].to(dtype=torch.float, device=device)
            # fft_spikes_noised_flatten = torch.flatten(fft_spikes_noised, 1)
            img_spikes = batch_spikes_data['img_spikes'].to(dtype=torch.float, device=device)
            img_spikes_noised = batch_spikes_data['img_spikes_noised'].to(dtype=torch.float, device=device)
            img_spikes_noised_norm = img_norm(img_spikes_noised)
            img_spikes_norm = img_norm(img_spikes)

            n_spikes = batch_spikes_data['n_spikes'].to(dtype=torch.float, device=device)
            n_spikes_emb = n_spikes / MAX_COUNT_SPIKES

            magnitude = fft_spikes_noised if noised_mag else fft_spikes
            # if not shift_fft:
            #     magnitude = torch.fft.fftshift(magnitude)
            img_spikes_pred = spikes_img_predictor(magnitude, n_spikes_emb)
            img_spikes_pred_norm = img_norm(img_spikes_pred)

            fft_spikes_pred = fft_magnitude(img_spikes_pred, shift=shift_fft)

            img_recon_loss = loss_recon_img_fun(img_spikes_norm, img_spikes_pred_norm)
            img_recon_sparsity = sparsity_f_loss(sigmoid(img_spikes_pred))
            img_recon_support_size = support_size_fun(img_spikes_pred)
            support_loss = support_loss_fun(n_spikes, img_recon_support_size)
            fft_spikes_norm = norm_mag(fft_spikes)
            fft_spikes_pred_norm = norm_mag(fft_spikes_pred)
            fft_recon_loss = loss_recon_fft_fun(fft_spikes_norm, fft_spikes_pred_norm)

            total_loss = lambda_img_recon * img_recon_loss + \
                         lambda_sparsity * img_recon_sparsity + \
                         lambda_fft_recon * fft_recon_loss + \
                         lambda_support_size * support_loss

            total_loss.backward()
            pred_optimizer.step()

            if ind % 1000 == 0:
                curr_lr = lr_scheduler.get_last_lr()
                log.info(f'epoch: {epoch}, ind: {ind}, '
                         f'img_recon_loss={img_recon_loss.detach().cpu().numpy()}, '
                         f'img_recon_sparsity={img_recon_sparsity.detach().cpu().numpy()}, '
                         f'fft_recon_loss={fft_recon_loss.detach().cpu().numpy()} ,'
                         f'support_loss={support_loss.detach().cpu().numpy()}'
                         f'loss_total: {total_loss.detach().cpu().numpy()}, '
                         f'lr={curr_lr}')
                total_loss_eval, img_recon_loss_eval, img_recon_sparsity_eval, fft_recon_loss_eval, support_loss_eval = eval_model()

                log.info(f'epoch: {epoch}, ind: {ind}, '
                         f'img_recon_loss_eval={img_recon_loss_eval}, '
                         f'img_recon_sparsity_eval={img_recon_sparsity_eval}, '
                         f'fft_recon_los_eval={fft_recon_loss_eval} ,'
                         f'support_loss_eval={support_loss_eval}, '
                         f'loss_total_eval: {total_loss_eval}')

                spikes_img_predictor.train()

            if ind % 5000 == 0:
                img_spikes_np = to_numpy_img(img_spikes, batch_size_eval)
                img_spikes_noised_np = to_numpy_img(img_spikes_noised, batch_size_eval)
                img_spikes_pred_np = to_numpy_img(img_spikes_pred, batch_size_eval)
                img_spikes_pred_merge = im_concatenate([img_spikes_np, img_spikes_noised_np, img_spikes_pred_np], 0)

                if not shift_fft:
                    fft_spikes = torch.fft.fftshift(fft_spikes)
                    fft_spikes_noised = torch.fft.fftshift(fft_spikes_noised)
                    fft_spikes_pred = torch.fft.fftshift(fft_spikes_pred)
                fft_spikes_np = to_numpy_img(fft_spikes, batch_size_eval)
                fft_spikes_noised_np = to_numpy_img(fft_spikes_noised, batch_size_eval)
                fft_spikes_pred_np = to_numpy_img(fft_spikes_pred, batch_size_eval)
                fft_spikes_pred_np_merge = im_concatenate([fft_spikes_np, fft_spikes_noised_np, fft_spikes_pred_np], 0)

                fft_spikes_ts = batch_data_test['fft_spikes'].to(dtype=torch.float, device=device)
                fft_spikes_noised_ts = batch_data_test['fft_spikes_noised'].to(dtype=torch.float, device=device)

                img_spikes_ts = batch_data_test['img_spikes'].to(dtype=torch.float, device=device)
                img_spikes_noised_ts = batch_data_test['img_spikes_noised'].to(dtype=torch.float, device=device)
                n_spikes_ts = batch_data_test['n_spikes'].to(dtype=torch.float, device=device)
                n_spikes_emb_ts = n_spikes_ts / MAX_COUNT_SPIKES

                spikes_img_predictor.eval()
                magnitude = fft_spikes_noised_ts if noised_mag else fft_spikes_ts
                img_spikes_pred_ts = spikes_img_predictor(magnitude, n_spikes_emb_ts)
                spikes_img_predictor.train()

                fft_spikes_pred_ts = fft_magnitude(img_spikes_pred_ts, shift=shift_fft)
                fft_spikes_pred_ts = fft_spikes_pred_ts

                img_spikes_np_ts = to_numpy_img(img_spikes_ts, 2 * batch_size_eval)
                img_spikes_noised_np_ts = to_numpy_img(img_spikes_noised_ts, 2 * batch_size_eval)
                img_spikes_pred_np_ts = to_numpy_img(img_spikes_pred_ts, 2 * batch_size_eval)
                img_spikes_pred_merge_ts = im_concatenate(
                    [img_spikes_np_ts, img_spikes_noised_np_ts, img_spikes_pred_np_ts], 0)

                fft_spikes_np_ts = to_numpy_img(torch.fft.fftshift(fft_spikes_ts), 2 * batch_size_eval)
                fft_spikes_noised_np_ts = to_numpy_img(torch.fft.fftshift(fft_spikes_noised_ts), 2 * batch_size_eval)
                fft_spikes_pred_np_ts = to_numpy_img(torch.fft.fftshift(fft_spikes_pred_ts), 2 * batch_size_eval)
                fft_spikes_pred_np_merge_ts = im_concatenate(
                    [fft_spikes_np_ts, fft_spikes_noised_np_ts, fft_spikes_pred_np_ts], 0)

                s3.save_object(os.path.join(out_save_dir, f'epoch_{epoch}', f'img_spikes_recon{epoch}-{ind}.jpg'),
                               im_save, img_spikes_pred_merge)
                s3.save_object(os.path.join(out_save_dir, f'epoch_{epoch}', f'fft_spikes_recon{epoch}-{ind}.jpg'),
                               im_save, fft_spikes_pred_np_merge)

                s3.save_object(os.path.join(out_save_dir, f'epoch_{epoch}', f'img_spikes_recon_ts_{epoch}-{ind}.jpg'),
                               im_save, img_spikes_pred_merge_ts)
                s3.save_object(os.path.join(out_save_dir, f'epoch_{epoch}', f'fft_spikes_recon_ts_{epoch}-{ind}.jpg'),
                               im_save, fft_spikes_pred_np_merge_ts)

        lr_scheduler.step()


def run_train_count_spikes(spikes_range: Tuple[int, int] = (2, 8),
                           batch_size: int = 64,
                           img_size: int = 32,
                           tile_size: int = 8,
                           noisy_domain: bool = False) -> nn.Module:
    log = logging.getLogger('SPIKES-COUNTER-TRAINER')

    def _eval_model():
        spikes_mlp_predictor.eval()
        pred_loss_eval = []
        acc_eval = []
        for ind in range(n_iters_eval):
            batch_spikes_data = next(iter(spike_sampler))
            if noisy_domain:
                x_fft_spikes = batch_spikes_data['fft_spikes_noised'].to(dtype=torch.float, device=device)
            else:
                x_fft_spikes = batch_spikes_data['fft_spikes'].to(dtype=torch.float, device=device)
            y = batch_spikes_data['n_spikes'].to(dtype=torch.float, device=device)
            y_pred = spikes_mlp_predictor(x_fft_spikes)
            pred_loss = loss_fun(y, y_pred)
            pred_loss_eval.append(pred_loss.detach().cpu().numpy())

            pred_spikes = torch.round(y_pred).to(torch.int).detach().cpu().numpy()
            n_spikes = y.to(torch.int).detach().cpu().numpy()
            acc = np.mean(pred_spikes == n_spikes)
            acc_eval.append(acc)

        pred_loss_eval = np.mean(np.array(pred_loss_eval))
        acc_eval = np.mean(np.array(acc_eval))

        y_pred_eval = y_pred.detach().cpu().numpy()
        pred_spikes_eval = torch.round(y_pred).to(torch.int).detach().cpu().numpy()
        n_spikes_eval = y.to(torch.int).detach().cpu().numpy()

        return pred_loss_eval, acc_eval, n_spikes_eval, y_pred_eval, pred_spikes_eval

    n_iters_tr = 50000
    n_iters_eval = 100
    gauss_noise = 0.0125

    lr = 0.00005
    loss_type = 'l1'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    log.info(f'img_size={img_size}, spikes_range={spikes_range}, n_iters={n_iters_tr}, '
             f'tile_size={tile_size}, noisy_domain: {noisy_domain}')

    spike_sampler = SpikesDataGenerator(spikes_range=spikes_range,
                                        img_size=img_size,
                                        batch_size=batch_size,
                                        add_gauss_noise=gauss_noise)

    spikes_mlp_predictor = SpikesCountPredictor(img_size, tile_size).to(device=device)

    log.info(f'{spikes_mlp_predictor}')

    spikes_mlp_predictor.train()

    if loss_type == 'l1':
        loss_fun = torch.nn.L1Loss()
    elif loss_type == 'l2':
        loss_fun = torch.nn.MSELoss()
    else:
        log.error(f'Not valid loss type: {loss_type}')

    pred_optimizer = optim.Adam(params=spikes_mlp_predictor.parameters(), lr=lr)

    for ind in tqdm(range(n_iters_tr)):
        pred_optimizer.zero_grad()
        batch_spikes_data = next(iter(spike_sampler))
        if noisy_domain:
            x = batch_spikes_data['fft_spikes_noised'].to(dtype=torch.float, device=device)
        else:
            x = batch_spikes_data['fft_spikes'].to(dtype=torch.float, device=device)
        y = batch_spikes_data['n_spikes'].to(dtype=torch.float, device=device)
        y_pred = spikes_mlp_predictor(x)
        pred_loss = loss_fun(y, y_pred)
        pred_loss.backward()
        pred_optimizer.step()

        pred_spikes = torch.round(y_pred).to(torch.int).detach().cpu().numpy()
        n_spikes = y.to(torch.int).detach().cpu().numpy()
        acc = np.mean(pred_spikes == n_spikes)

        if ind % 500 == 0:
            log.info(f'ind: {ind}, loss: {pred_loss.detach().cpu().numpy()}, acc = {acc}')

        if ind % 2000 == 0:
            pred_loss_eval, acc_eval, _, _, _ = _eval_model()
            spikes_mlp_predictor.train()
            log.info(f'ind: {ind}, pred_loss_eval={pred_loss_eval}, acc_eval:{acc_eval}')

    pred_loss_eval, acc_eval, n_spikes_eval, y_pred_eval, pred_spikes_eval = _eval_model()
    log.info(f'FINAL pred_loss_eva={pred_loss_eval}, acc_eva:{acc_eval}')

    for n_spikes_, n_spikes_pred_, y_pred_ in zip(n_spikes_eval, pred_spikes_eval, y_pred_eval):
        err_smbl = 'V' if n_spikes_ == n_spikes_pred_ else 'X'
        log.info(f'{n_spikes_} -{err_smbl}- {n_spikes_pred_}, y_out: {y_pred_}')

    return spikes_mlp_predictor


if __name__ == '__main__':
    fire.Fire()
