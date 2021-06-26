from torch import Tensor
import torch.nn as nn
from models.layers import ConvBlock, UpConvBlock, DownConvBlock


# class ConvUnetPhaseRetrieval(UNetConv):
#     def __init__(self):
#         super(ConvUnetPhaseRetrieval, self).__init__(img_ch=1, output_ch=2)
#
#     def forward(self, x: Tensor) -> Tensor:
#         b, c,  h, w = x.shape
#         x = torch.fft.fftshift(x, dim=(-2, -1))
#         x = super(ConvUnetPhaseRetrieval, self).forward(x)
#         x = torch.view_as_complex(x.permute(0, 2, 3, 1).reshape(-1, 2))
#         x = x.view(b, c, h, w)
#         x = torch.fft.ifftshift(x, dim=(-2, -1))
#         return x
#
#
# class ConvUnetPRanglePred(UNetConv):
#     def __init__(self):
#         super(ConvUnetPRanglePred, self).__init__(img_ch=1, output_ch=1)
#
#     def forward(self, x: Tensor) -> Tensor:
#         x = torch.fft.fftshift(x, dim=(-2, -1))
#         angle_pred = super(ConvUnetPRanglePred, self).forward(x)
#         x = x * torch.exp(torch.view_as_complex(torch.stack([torch.zeros_like(x), angle_pred], -1)))
#         x = torch.fft.ifftshift(x, dim=(-2, -1))
#         return x


class EncoderConv(nn.Module):
    def __init__(self, in_ch=1, encoder_ch=8, deep: int = 3, last_down: bool = True):
        super(EncoderConv, self).__init__()
        self.encoder_ch = encoder_ch
        self.deep = deep

        self.conv_down_blocks = []
        inp_ch_block = in_ch
        self.out_ch = encoder_ch
        for ind_block in range(self.deep):

            if ind_block == self.deep - 1 and last_down:
                conv_block = ConvBlock(ch_in=inp_ch_block, ch_out=self.out_ch)
            else:
                conv_block = DownConvBlock(ch_in=inp_ch_block, ch_out=self.out_ch)
            inp_ch_block = self.out_ch

            if ind_block < self.deep - 1:
                self.out_ch *= 2

            self.conv_down_blocks.append(conv_block)
        self.conv_down_blocks = nn.Sequential(*self.conv_down_blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv_down_blocks(x)


class DecoderConv(nn.Module):
    def __init__(self, img_ch: int = 32, output_ch: int = 1, deep: int = 3):
        super(DecoderConv, self).__init__()
        # assert deep == 3
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
                conv_block = ConvBlock(ch_out, ch_out)

            ch_im = ch_out
            self.conv_blocks.append(conv_block)
        self.conv_blocks = nn.Sequential(*self.conv_blocks)
        #
        # up_conv1 = UpConvBlock(ch_in=img_ch, ch_out=img_ch // 2)   # 32x14x14
        # conv1 = ConvBlock(ch_in=img_ch // 2, ch_out=img_ch // 2)  # 32x14x14
        #
        # up_conv2 = UpConvBlock(ch_in=img_ch // 2, ch_out=img_ch // 4)  # 16x28x28
        # conv2 = ConvBlock(ch_in=img_ch // 4, ch_out=img_ch // 4)  # 16x28x28
        #
        # conv_out = nn.Conv2d(img_ch // 4, output_ch, kernel_size=1, stride=1, padding=0)  # 1x28x28
        #
        # self.conv_up_blocks = nn.Sequential(up_conv1, conv1, up_conv2, conv2, conv_out)

    def forward(self, x: Tensor) -> Tensor:
        # decoding path
        return self.conv_blocks(x)
        # return self.conv_up_blocks(x)


