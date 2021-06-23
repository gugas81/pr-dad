
import torch
import numpy as np
from torch import Tensor
import torch.optim as optim
import fire
import matplotlib
import os

from models.net_models import ConvUnetPhaseRetrieval, PhaseRetrievalPredictor, ConvUnetPRanglePred, UNetConv

from common import LossesPRImages, InferredBatch
from common import im_save, im_concatenate, plot_losses_metrics

from training.base_phase_retrieval_trainer import TrainerPhaseRetrieval


class TrainerPhaseRetrievalImages(TrainerPhaseRetrieval):
    def __init__(self, debug_dir: str = None, use_ref_net: bool = False, recon_type: str = 'fc', multy_coeff: int = 1):
        super(TrainerPhaseRetrievalImages, self).__init__(debug_dir=debug_dir)

        self.use_ref_net = use_ref_net

        if recon_type == 'convnet':
            self.recon_net = ConvUnetPhaseRetrieval()
        elif recon_type == 'fc':
            self.recon_net = PhaseRetrievalPredictor(multy_coeff=multy_coeff)
        elif recon_type == 'angle-pred':
            self.recon_net = ConvUnetPRanglePred()

        self.recon_net.train()
        self.recon_net.to(device=self.device)

        self.params = list(self.recon_net.parameters())
        if self.use_ref_net:
            self.unet_ref = UNetConv()
            self.unet_ref.train()
            self.unet_ref.to(device=self.device)
            self.params += list(self.unet_ref.parameters())
        else:
            self.unet_ref = None

        self.optimizer = optim.Adam(params=self.params, lr=self.learning_rate)

    def train_epoch(self, epoch: int) -> LossesPRImages:
        train_losses = []
        if epoch < 5:
            alpha_blend = epoch * 0.1
        else:
            alpha_blend = 0.5

        for batch_idx, (data_batch, labels_batch) in enumerate(self.train_loader):
            data_batch = data_batch.to(self.device)
            self.optimizer.zero_grad()

            inferred_batch = self.forward(data_batch, alpha_blend=alpha_blend)
            losses_batch = self.calc_losses(data_batch, inferred_batch)
            losses_batch.total.backward()
            self.optimizer.step()

            if batch_idx % self.log_interval == 0:
                train_losses.append(losses_batch.detach())
                print(f'Train Epoch: {epoch} [{batch_idx * len(data_batch)}/{len(self.train_loader.dataset)}'
                      f'({100. * batch_idx / len(self.train_loader):.0f}%)], {losses_batch}')

        train_losses = LossesPRImages.merge(train_losses)
        return train_losses

    def test_eval(self) -> LossesPRImages:
        losses = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data = data.to(self.device)
                inferred_batch = self.forward(data)
                losses_batch = self.calc_losses(data, inferred_batch)
                losses.append(losses_batch)

        return LossesPRImages.merge(losses).mean()

    def calc_losses(self, data: Tensor, inferred_batch: InferredBatch) -> LossesPRImages:
        mse_img = self.l2_loss(inferred_batch.img_recon.real, data)
        mse_magnitude = self.l2_loss(inferred_batch.fft_magnitude_recon, inferred_batch.fft_magnitude)
        l2norm_imag_part = torch.mean(torch.square(inferred_batch.img_recon.imag.abs()))
        l1norm_real_part = torch.mean(inferred_batch.img_recon.real.abs())
        l1norm_orig = torch.mean(data.abs())

        total = mse_img + mse_magnitude + l1norm_real_part
        losses = LossesPRImages(l2norm_imag_part=l2norm_imag_part,
                                mse_magnitude=mse_magnitude,
                                l1norm_real_part=l1norm_real_part,
                                l1_orig=l1norm_orig,
                                mse_img=mse_img)

        if self.use_ref_net:
            losses.l1norm_real_part_ref = torch.mean(inferred_batch.img_recon_ref.abs())
            losses.mse_img_ref = self.l2_loss(inferred_batch.img_recon_ref, data)
            losses.mse_magnitude_ref = self.l2_loss(inferred_batch.fft_magnitude_recon_ref, inferred_batch.fft_magnitude)
            total += losses.mse_img_ref + losses.mse_magnitude_ref + losses.l1norm_real_part_ref

        losses.total = total
        return losses

    def forward(self, data_batch: Tensor, alpha_blend: float = 1.0) -> InferredBatch:
        fft_magnitude = self.forward_magnitude_fft(data_batch)
        fft_recon = self.recon_net(fft_magnitude)
        fft_magnitude_recon = fft_recon.abs()
        img_recon = torch.fft.ifft2(fft_recon, norm=self._fft_norm)
        inferred_batch = InferredBatch(fft_magnitude=fft_magnitude,
                                       fft_magnitude_recon=fft_magnitude_recon,
                                       img_recon=img_recon)
        if self.use_ref_net:
            recon_blend_img = alpha_blend * img_recon.real + (1-alpha_blend) * data_batch
            inferred_batch.img_recon_ref = self.unet_ref(recon_blend_img)
            inferred_batch.fft_magnitude_recon_ref = torch.fft.fft2(fft_recon, norm=self._fft_norm).abs()

        return inferred_batch

    def _save_debug_imgs(self, save_dir: str, postifx: str = ''):
        with torch.no_grad():
            dbg_imgs_batch_ts = self._get_dbg_imgs(self.data_ts_batch)
            dbg_imgs_batch_tr = self._get_dbg_imgs(self.data_tr_batch)

        im_save(dbg_imgs_batch_tr, os.path.join(save_dir, f'tr-batch-{postifx}.jpg'))
        im_save(dbg_imgs_batch_ts, os.path.join(save_dir, f'ts-batch-{postifx}.jpg'))

    def _get_dbg_imgs(self, data_batch_np):
        def norm_img(img_arr):
            img_arr -= img_arr.min()
            img_arr /= max(img_arr.max(), np.finfo(img_arr.dtype).eps)
            return img_arr
        inferred_batch_ts = self.forward(data_batch_np)
        data_predict_batch = np.transpose(inferred_batch_ts.img_recon.real.detach().cpu().numpy(),
                                             axes=(0, 2, 3, 1))
        data_batch_np = norm_img(np.transpose(data_batch_np.detach().cpu().numpy(), axes=(0, 2, 3, 1)))
        data_batch_np = im_concatenate(data_batch_np)
        data_predict_batch = norm_img(im_concatenate(data_predict_batch))
        data_dbg_batch = [data_batch_np, data_predict_batch]

        if self.use_ref_net:
            data_predict_batch_ref = np.transpose(inferred_batch_ts.img_recon_ref.detach().cpu().numpy(),
                                                  axes=(0, 2, 3, 1))
            data_predict_batch_ref = norm_img(im_concatenate(data_predict_batch_ref))
            data_dbg_batch += [data_predict_batch_ref]

        data_dbg_batch = im_concatenate(data_dbg_batch, axis=0)
        return data_dbg_batch

    def run_train(self) -> (LossesPRImages, LossesPRImages):
        train_losses = []
        test_losses = []
        with torch.no_grad():
            init_test_loss = self.test_eval()
        print(f'Init test eval: {init_test_loss}')
        if self.debug_dir:
            self._save_debug_imgs(self.debug_dir, "0")

        for epoch in range(1, self.n_epochs + 1):
            train_losses_batch = self.train_epoch(epoch)
            train_losses.append(train_losses_batch)
            if self.debug_dir:
                self._save_debug_imgs(self.debug_dir, str(epoch))
            with torch.no_grad():
                test_losses_batch = self.test_eval()
                test_losses.append(test_losses_batch)
                print(f'TEST EVAL epoch={epoch}: {test_losses_batch}')
        with torch.no_grad():
            train_losses = LossesPRImages.merge(train_losses)
            test_losses = LossesPRImages.merge(test_losses)
        return train_losses, test_losses

def run_trainer_PhaseRetrievalImage(experiment_name: str = 'recon-l2-imgs-adam-optim',
                                    use_ref_net: bool = False,
                                    recon_type: str = 'fc',
                                    multy_coeff: int = 1):
    import matplotlib.pyplot as plt

    matplotlib.use('module://backend_interagg')
    debug_root = '/home/ubuntu/code/phase-retrieval/dbg-pr-images'
    debug_dir = os.path.join(debug_root, experiment_name)
    os.makedirs(debug_dir, exist_ok=True)
    trainer = TrainerPhaseRetrievalImages(debug_dir=debug_dir, use_ref_net=use_ref_net, recon_type=recon_type,
                                          multy_coeff=multy_coeff)

    data_batch = iter(trainer.test_loader).next()[0].to(device=trainer.device)

    plt.imshow(data_batch[0][0].detach().cpu().numpy(), cmap='gray', interpolation='none')
    plt.show()
    train_losses, test_losses = trainer.run_train()
    inferred_batch = trainer.forward(data_batch)
    plt.imshow(inferred_batch.img_recon.real[0][0].detach().cpu().numpy(), cmap='gray', interpolation='none')
    plt.show()
    if inferred_batch.img_recon_ref is not None:
        plt.imshow(inferred_batch.img_recon_ref[0][0].detach().cpu().numpy(), cmap='gray', interpolation='none')
        plt.show()

    plot_losses_metrics(train_losses, 'train_losses')
    plot_losses_metrics(test_losses, 'test_losses')


if __name__ == '__main__':
    fire.Fire(run_trainer_PhaseRetrievalImage)
