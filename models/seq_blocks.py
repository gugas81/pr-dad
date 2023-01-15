import numpy as np
from torch import Tensor
import torch.nn as nn
from models.layers import ConvBlock, UpConvBlock, DownConvBlock, ResBlock, FcBlock
from typing import Optional, List, Union
from models.untils import BlockList


class MlpNet(nn.Module):
    def __init__(self,
                 in_ch: int,
                 deep: int,
                 out_ch: Optional[int] = None,
                 ch_list: List[int] = None,
                 use_dropout: bool = False,
                 multy_coeff: float = 2.0,
                 norm_type: str = None,
                 active_type: str = 'leakly_relu',
                 active_ch: bool = True):
        super(MlpNet, self).__init__()
        self.fc_layers = BlockList()

        if ch_list is None or len(ch_list) == 0:
            ch_list = [in_ch]
            for ind_block in range(1, deep, 1):
                if ind_block == deep - 1 and out_ch:

                    ch_list.append(out_ch if ind_block == deep - 1 and out_ch else ch_list[ind_block-1] * multy_coeff)

        for ind_block in range(deep-1):
            in_ch, out_ch = ch_list[ind_block], ch_list[ind_block+1]
            fc_block = FcBlock(in_features=in_ch,
                               out_features=out_ch,
                               use_dropout=use_dropout,
                               norm_type=norm_type,
                               active_type=active_type,
                               active_params=out_ch if active_ch else 1)
            self.fc_layers.append(fc_block)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc_layers(x)


class EncoderConv(nn.Module):
    def __init__(self, in_ch=1, encoder_ch=8, deep: int = 3, last_down: bool = True, use_res_blocks: bool = False,
                 out_ch: Optional[int] = None,
                 down_pool: str = 'avrg_pool',
                 active_type: str = 'leakly_relu',
                 padding_mode: str = 'zeros'):
        super(EncoderConv, self).__init__()
        self.encoder_ch = encoder_ch
        self.deep = deep

        self.conv_down_blocks = BlockList()
        inp_ch_block = in_ch
        self.out_ch = []
        curr_out_ch = encoder_ch
        for ind_block in range(self.deep):
            if ind_block == self.deep - 1 and out_ch:
                curr_out_ch = out_ch

            self.out_ch.append(curr_out_ch)
            if ind_block == 0 or (ind_block == self.deep - 1 and not last_down):
                # if use_res_blocks:
                #     conv_block = ResBlock(in_channels=inp_ch_block, out_channels=self.out_ch)
                # else:
                conv_block = ConvBlock(ch_in=inp_ch_block, ch_out=curr_out_ch, active_type=active_type, padding_mode=padding_mode)
            else:
                conv_block = DownConvBlock(ch_in=inp_ch_block, ch_out=curr_out_ch,
                                           use_res_block=use_res_blocks, active_type=active_type, down_pool=down_pool)
            inp_ch_block = curr_out_ch

            if ind_block < self.deep - 1:
                curr_out_ch *= 2

            self.conv_down_blocks.append(conv_block)

    def forward(self, x: Tensor, use_residual: bool = False) -> Union[Tensor, List[Tensor]]:
        return self.conv_down_blocks(x, use_residual=use_residual)

    def get_layers(self) -> BlockList:
        return self.conv_down_blocks


class DecoderConv(nn.Module):
    def __init__(self, img_ch: int = 32, output_ch: Optional[int] = None, deep: int = 3,
                 use_res_blocks: bool = False, skip_connect_ch:  List[int] = None, up_mode: str = 'nearest',
                 active_type: str = 'leakly_relu'):
        super(DecoderConv, self).__init__()
        self.deep = deep
        self.conv_layers = BlockList()
        self.out_layers = BlockList()
        ch_im = img_ch
        ch_out = ch_im
        self.ch_outs = []
        if skip_connect_ch is None:
            skip_connect_ch = np.zeros(self.deep, dtype=np.int)
        for ind_layer in range(self.deep):
            ch_im += skip_connect_ch[ind_layer]
            if ind_layer == self.deep - 1:
                if output_ch is not None and not self.multi_scale_out:
                    conv_layer = nn.Conv2d(ch_im, output_ch, kernel_size=1, stride=1, padding=0)
                else:
                    ch_out = ch_out // 2
                    if use_res_blocks:
                        conv_layer = ResBlock(ch_im, ch_out)
                    else:
                        conv_layer = ConvBlock(ch_im, ch_out)
            else:
                ch_out = ch_out // 2
                up_conv_block = UpConvBlock(ch_in=ch_im, ch_out=ch_out, up_mode=up_mode, active_type=active_type)
                if use_res_blocks:
                    conv_block = ResBlock(ch_out, ch_out, active_type=active_type)
                else:
                    conv_block = ConvBlock(ch_out, ch_out,  active_type=active_type)
                conv_layer = nn.Sequential(up_conv_block, conv_block)

            ch_im = ch_out
            self.ch_outs.append(ch_out)
            self.conv_layers.append(conv_layer)

    def get_layers(self) -> BlockList:
        return self.conv_layers

    def forward(self, x: Tensor, use_residual: bool = False) -> Union[Tensor, List[Tensor]]:
        return self.conv_layers(x, use_residual=use_residual)




