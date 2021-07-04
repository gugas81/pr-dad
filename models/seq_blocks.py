from torch import Tensor
import torch.nn as nn
from models.layers import ConvBlock, UpConvBlock, DownConvBlock, ResBlock


class EncoderConv(nn.Module):
    def __init__(self, in_ch=1, encoder_ch=8, deep: int = 3, last_down: bool = True, use_res_blocks: bool = False):
        super(EncoderConv, self).__init__()
        self.encoder_ch = encoder_ch
        self.deep = deep

        self.conv_down_blocks = []
        inp_ch_block = in_ch
        self.out_ch = encoder_ch
        for ind_block in range(self.deep):
            if ind_block == self.deep - 1 and last_down:
                if use_res_blocks:
                    conv_block = ResBlock(in_channels=inp_ch_block, out_channels=self.out_ch)
                else:
                    conv_block = ConvBlock(ch_in=inp_ch_block, ch_out=self.out_ch)
            else:
                conv_block = DownConvBlock(ch_in=inp_ch_block, ch_out=self.out_ch, use_res_block=use_res_blocks)
            inp_ch_block = self.out_ch

            if ind_block < self.deep - 1:
                self.out_ch *= 2

            self.conv_down_blocks.append(conv_block)
        self.conv_down_blocks = nn.Sequential(*self.conv_down_blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv_down_blocks(x)


class DecoderConv(nn.Module):
    def __init__(self, img_ch: int = 32, output_ch: int = 1, deep: int = 3, use_res_blocks: bool = False):
        super(DecoderConv, self).__init__()
        self.deep = deep
        self.conv_blocks = []
        ch_im = img_ch
        for ind_block in range(self.deep):

            if ind_block == self.deep - 1:
                conv_block = nn.Conv2d(ch_out, output_ch, kernel_size=1, stride=1, padding=0)
            else:
                ch_out = ch_im // 2
                up_conv_block = UpConvBlock(ch_in=ch_im, ch_out=ch_out)
                self.conv_blocks.append(up_conv_block)
                if use_res_blocks:
                    conv_block = ResBlock(ch_out, ch_out)
                else:
                    conv_block = ConvBlock(ch_out, ch_out)

            ch_im = ch_out
            self.conv_blocks.append(conv_block)
        self.conv_blocks = nn.Sequential(*self.conv_blocks)

    def forward(self, x: Tensor) -> Tensor:
        # decoding path
        return self.conv_blocks(x)


