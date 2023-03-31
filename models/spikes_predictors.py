from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torchvision import transforms

from models import MlpNet, UNetConv, AeConv

_EPSILON = 1e-8


def proj_magnitude(img_x, img_mag, shifted: bool = True, norm="forward"):
    fft_img = torch.fft.fft2(img_x, norm=norm)
    if shifted:
        img_mag = torch.fft.fftshift(img_mag)
    fft_img_changed = img_mag * fft_img / (torch.abs(fft_img) + _EPSILON)
    img_changed = torch.fft.ifft2(fft_img_changed, norm=norm).real
    return img_changed


class SpikesImgReconConvModel(nn.Module):
    def __init__(self,
                 img_size: int,
                 spikes_meta_size: int = 0,
                 count_spikes_emb_sie: int = 8,
                 tile_size: int = None,
                 is_proj_mag: bool = False,
                 pred_type: str = 'conv_ae',
                 conv_net_deep: int = 3,
                 n_encoder_ch: int = 16,
                 count_predictor: bool = False,
                 multi_scale_out: bool = False,
                 deep_backbone_map: int =0):
        super(SpikesImgReconConvModel, self).__init__()
        self._pred_type = pred_type
        if pred_type == 'conv_unet':
            self._conv_model = UNetConv(up_mode='bilinear', img_size=img_size, deep=conv_net_deep)
        elif pred_type == 'conv_ae':
            self._conv_model = AeConv(img_size=img_size,
                                      deep=conv_net_deep,
                                      n_encoder_ch=n_encoder_ch,
                                      multi_scale_out=multi_scale_out,
                                      deep_backbone_map=deep_backbone_map)
        else:
            raise TypeError(f'unknown conv type: {pred_type}')

        if count_predictor:
            n_features = self._conv_model.n_enc_features_ch
            n_features_size = self._conv_model.n_features_size
            size_flatten_in = n_features * (n_features_size ** 2)
            ch_list = [size_flatten_in,
                       size_flatten_in // 4, size_flatten_in // 4,
                       size_flatten_in // 8, size_flatten_in // 8,
                       size_flatten_in // 16, size_flatten_in // 16, 1]

            self._count_spikes_predictor = MlpNet(in_ch=size_flatten_in,
                                                  ch_list=ch_list,
                                                  out_ch=1,
                                                  deep=len(ch_list),
                                                  multy_coeff=0.5)
        else:
            self._count_spikes_predictor = None

    def forward(self, magnitude: Tensor, meta_x: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:

        x_out, enc_features, dec_features, coeff = self._conv_model(magnitude)
        recon_img = x_out #.squeeze(1)

        if self._count_spikes_predictor is not None:
            enc_features_flatten = torch.flatten(enc_features, 1)
            n_spikes_pred = self._count_spikes_predictor(enc_features_flatten)
        else:
            n_spikes_pred = None

        return recon_img, n_spikes_pred


class SpikesImgReconMlpModel(nn.Module):
    def __init__(self,
                 img_size: int,
                 spikes_meta_size: int = 0,
                 count_spikes_emb_sie: int = 8,
                 tile_size: int = None,
                 is_proj_mag: bool = False,
                 fft_shifted: bool = False):
        super(SpikesImgReconMlpModel, self).__init__()

        self._img_size = img_size
        self._is_proj_mag = is_proj_mag
        self._fft_shifted = fft_shifted
        self._tile_size = self._img_size if tile_size is None else tile_size

        size_flatten_in = self._tile_size ** 2
        size_flatten_out = self._img_size ** 2

        self._spikes_meta_size = spikes_meta_size
        self._count_spikes_emb_sie = count_spikes_emb_sie

        ch_list_encoder = [size_flatten_in,
                           size_flatten_in // 2, size_flatten_in // 2,
                           size_flatten_in // 4, size_flatten_in // 4,
                           size_flatten_in // 8, size_flatten_in // 8]

        bottle_neck = ch_list_encoder[-1] + self._count_spikes_emb_sie
        ch_list_decoder = [bottle_neck, bottle_neck,
                           size_flatten_out // 4, size_flatten_out // 4,
                           size_flatten_out // 2, size_flatten_out // 2,
                           size_flatten_out]
        ch_list_embeder = [self._spikes_meta_size, self._count_spikes_emb_sie // 2, self._count_spikes_emb_sie]

        self._mlp_predictor_encoder = MlpNet(in_ch=size_flatten_in,
                                             ch_list=ch_list_encoder,
                                             deep=len(ch_list_encoder))

        self._crop_tile = transforms.CenterCrop(self._tile_size)

        if self._spikes_meta_size > 0:
            self._mlp_emdeder = MlpNet(in_ch=self._spikes_meta_size,
                                       ch_list=ch_list_embeder,
                                       deep=len(ch_list_embeder))
        else:
            self._mlp_emdeder = nn.Identity()

        self._mlp_predictor_decoder = MlpNet(in_ch=bottle_neck,
                                             ch_list=ch_list_decoder,
                                             out_ch=size_flatten_out,
                                             deep=len(ch_list_decoder))

    def forward(self, magnitude: Tensor, meta_x: Optional[Tensor] = None) -> Tensor:
        magnitude_tile = self._crop_tile(magnitude)
        magnitude_flatten = torch.flatten(magnitude_tile, 1)
        encode = self._mlp_predictor_encoder(magnitude_flatten)
        batch_size = magnitude.shape[0]

        if meta_x is not None:
            emb = self._mlp_emdeder(meta_x)
            encode = torch.cat([encode, emb], dim=1)

        img_recon = self._mlp_predictor_decoder(encode).view(batch_size, 1, self._img_size, self._img_size)

        if self._is_proj_mag:
            img_recon = proj_magnitude(img_recon, magnitude, shifted=self._fft_shifted)

        return img_recon


class SpikesCountPredictor(nn.Module):
    def __init__(self, img_size: int, tile_size: int = None):
        super(SpikesCountPredictor, self).__init__()
        self._img_size = img_size
        self._tile_size = self._img_size if tile_size is None else tile_size

        size_flatten_in = self._tile_size ** 2  # (1 + self._tile_size // 2)

        ch_list = [size_flatten_in,
                   size_flatten_in // 2, size_flatten_in // 2,
                   size_flatten_in // 4, size_flatten_in // 4,
                   size_flatten_in // 8, size_flatten_in // 8, 1]

        self._mlp_predictor = MlpNet(in_ch=self._tile_size,
                                     ch_list=ch_list,
                                     out_ch=1,
                                     deep=len(ch_list),
                                     multy_coeff=0.5)

        self._crop_tile = transforms.CenterCrop(self._tile_size) if self._img_size != self._tile_size else nn.Identity()

    def forward(self, magnitude: Tensor) -> Tensor:
        magnitude_tile = self._crop_tile(magnitude)  # [..., :(1 + self._tile_size // 2)]
        magnitude_flatten = torch.flatten(magnitude_tile, 1)
        pred_count_spikes = self._mlp_predictor(magnitude_flatten)

        return pred_count_spikes
