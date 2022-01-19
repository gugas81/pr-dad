import torch
import torch.nn as nn
import math
import os
import pickle
from torch import Tensor

from pytorch_wavelets import DWTForward, DWTInverse

from common import PATHS

from models.base_autoencoder import BaseAe
WAVELET_HAAR_WEIGHTS_PATH = os.path.join(PATHS.PROJECT_ROOT, 'models', 'wavelet_haar_weights_c2.pkl')


class WaveletHaarTransform(nn.Module):
    def __init__(self,
                 in_ch: int = 1,
                 scale: int = 1,
                 mode='reflect',
                 decomposition: bool = True,
                 transpose=True,
                 filters_path: str = WAVELET_HAAR_WEIGHTS_PATH):
        super(WaveletHaarTransform, self).__init__()
        assert in_ch == 1 or in_ch == 3
        self.scale = scale
        self.decomposition = decomposition
        self.transpose = transpose
        self.in_ch = in_ch

        self.kernel_size = int(math.pow(2, self.scale))
        self.subband_channels = in_ch * self.kernel_size * self.kernel_size

        if self.decomposition:
            self.wavelet_conv_transform = nn.Conv2d(in_channels=in_ch,
                                                    out_channels=self.subband_channels,
                                                    kernel_size=self.kernel_size,
                                                    stride=self.kernel_size,
                                                    padding=0,
                                                    groups=in_ch,
                                                    padding_mode=mode,
                                                    bias=False)
        else:
            self.wavelet_conv_transform = nn.ConvTranspose2d(in_channels=self.subband_channels,
                                                             out_channels=in_ch,
                                                             kernel_size=self.kernel_size,
                                                             stride=self.kernel_size,
                                                             padding=0,
                                                             groups=in_ch,
                                                             padding_mode=mode,
                                                             bias=False)

        self._init_filters(filters_path)

    def extra_repr(self) -> str:
        return f'decomposition={self.decomposition}, ' \
               f'transpose={self.transpose}' \
               f'scale={self.scale}, ' \
               f'img_channels={self.in_ch},  ' \
               f'subband_channels={self.subband_channels}, ' \
               f'kernel_size={self.kernel_size}'

    def _init_filters(self, params_path: str):
        assert os.path.isfile(params_path)
        with open(params_path, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            conv_wavelets_weights = u.load()
        conv_filter = torch.from_numpy(conv_wavelets_weights['rec%d' % self.kernel_size])[:self.subband_channels]
        for submodules in self.modules():
            if isinstance(submodules, nn.Conv2d) or isinstance(submodules, nn.ConvTranspose2d):
                submodules.weight.data = conv_filter
                submodules.weight.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        if self.decomposition:
            output = self.wavelet_conv_transform(x)
            if self.transpose:
                ous_size = output.size()
                output = output.view(ous_size[0], self.in_ch, -1, ous_size[2], ous_size[3]).transpose(1, 2).contiguous().view(
                    ous_size)
        else:
            if self.transpose:
                xx = x
                in_size = xx.size()
                xx = xx.view(in_size[0], -1, self.in_ch, in_size[2], in_size[3]).transpose(1, 2).contiguous().view(in_size)
            output = self.wavelet_conv_transform(xx)
        return output


class ForwardWaveletSubbandsTransform(nn.Module):
    def __init__(self, imgs_size: int, in_ch: int = 1, scale=3, mode='reflect', wave='db3',  ordered: bool = False):
        super(ForwardWaveletSubbandsTransform, self).__init__()
        assert in_ch == 1
        self._j = scale
        self.wave = wave
        self.mode = mode
        self._in_ch = in_ch
        self._ordered = ordered
        self._dwt_j1 = DWTForward(J=1, mode=mode, wave=wave)
        self._imgs_size = imgs_size

    def extra_repr(self) -> str:
        return f'scale={self._j}, ' \
               f'img_size:={self._imgs_size}' \
               f'img_channels={self._in_ch},  ' \
               f'wave={self.wave}, ' \
               f'ordered={self._ordered}' \
               f'pad_mode: {self.mode}'

    def forward(self, x: Tensor) -> Tensor:
        subbands = x
        for _ in range(self._j):
            y_l, y_h = self._dwt_j1(subbands)
            y_h = y_h[0]
            batch, c, subb, h, w = y_h.shape
            if self._ordered:
                y_h = torch.permute(y_h, (0, 2, 1, 3, 4))
            y_h = y_h.reshape(batch, c * subb, h, w)
            subbands = torch.cat([y_l, y_h], dim=1)

        return subbands


class InverseWaveletSubbandsTransform(nn.Module):
    def __init__(self, imgs_size: int, in_ch: int = 1, scale=3, mode='reflect', wave='db3', ordered: bool = False):
        super(InverseWaveletSubbandsTransform, self).__init__()
        assert in_ch == 1
        self._j = scale
        self.wave = wave
        self.mode = mode
        self._in_ch = in_ch
        self._ordered = ordered
        self._idwt = DWTInverse(mode=mode, wave=wave)
        self._imgs_size = imgs_size

    def extra_repr(self) -> str:
        return f'scale={self._j}, ' \
               f'img_size:={self._imgs_size}' \
               f'img_channels={self._in_ch},  ' \
               f'wave={self.wave}, ' \
               f'ordered={self._ordered}' \
               f'pad_mode: {self.mode}'

    def forward(self, x: Tensor) -> Tensor:
        recon_x = x
        for _ in range(self._j):
            b, c, h, w = recon_x.shape
            n_s = c // 4
            y_l = recon_x[:, :n_s]
            if self._ordered:
                    y_h = recon_x[:, n_s:].view(b, 3, n_s, h, w)
                    y_h = torch.permute(y_h, (0, 2, 1, 3, 4))
            else:
                y_h = recon_x[:, n_s:].view(b, n_s, 3, h, w)
            recon_x = self._idwt((y_l, [y_h]))
        recon_x = recon_x[:, :, :self._imgs_size, :self._imgs_size]
        return recon_x


class WaveletTransformAe(BaseAe):
    def __init__(self, img_size: int, in_ch: int = 1, deep: int = 3, mode: str = 'reflect', wave: str = 'db3',
                 norm_ds: bool = False):
        super(WaveletTransformAe, self).__init__(img_size=img_size, deep=deep, in_ch=in_ch)
        self._norm_ds = norm_ds
        self._deep = deep
        if wave == 'haar':
            self._encoder = WaveletHaarTransform(in_ch=in_ch,
                                                 scale=deep,
                                                 decomposition=True,
                                                 mode=mode)

            self._decoder = WaveletHaarTransform(in_ch=in_ch,
                                                 scale=deep,
                                                 decomposition=False,
                                                 mode=mode)
        else:
            self._encoder = ForwardWaveletSubbandsTransform(imgs_size=img_size,
                                                            in_ch=in_ch,
                                                            scale=deep,
                                                            mode=mode,
                                                            wave=wave)

            self._decoder = InverseWaveletSubbandsTransform(imgs_size=img_size,
                                                            in_ch=in_ch,
                                                            scale=deep,
                                                            mode=mode,
                                                            wave=wave)
        dummy_input = torch.zeros((1, 1, img_size, img_size))
        dummy_features = self.encode(dummy_input)
        _, self._n_enc_features_ch, self._n_features_size, n_features_size_y = dummy_features.shape

    @property
    def n_enc_features_ch(self):
        return self._n_enc_features_ch

    @property
    def n_dec_features_ch(self) -> int:
        return self._n_enc_features_ch

    @property
    def n_features_size(self):
        return self._n_features_size

    def encode(self, x: Tensor) -> Tensor:
        features = self._encoder(x)
        if self._norm_ds:
            features[:, 0] = (1 / self._deep) * features[:, 0]
        return features

    def decode(self, features: Tensor) -> Tensor:
        if self._norm_ds:
            features[:, 0] = self._deep * features[:, 0]
        x_out = self._decoder(features)
        return x_out

    def forward(self, x: Tensor) -> (Tensor, Tensor, Tensor, Tensor):
        enc_features = self.encode(x)
        dec_features = enc_features
        coeff = enc_features
        x_out = self.decode(dec_features)

        return x_out, enc_features, dec_features, coeff
