import torch
from torch import Tensor
import torch.nn as nn
from models.layers import ConvBlock, UpConvBlock


class UNetConv(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(UNetConv, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_1 = ConvBlock(ch_in=img_ch, ch_out=16)
        self.conv_2 = ConvBlock(ch_in=16, ch_out=32)
        self.conv_3 = ConvBlock(ch_in=32, ch_out=64)

        self.up_3 = UpConvBlock(ch_in=64, ch_out=32)
        self.up_conv_3 = ConvBlock(ch_in=64, ch_out=32)

        self.up_2 = UpConvBlock(ch_in=32, ch_out=16)
        self.up_conv_2 = ConvBlock(ch_in=32, ch_out=16)

        self.conv_out = nn.Conv2d(16, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        # encoding path
        x1 = self.conv_1(x) # 16x27x27

        x2 = self.maxpool(x1)# 16x14x14
        x2 = self.conv_2(x2) # 32x14x14

        x3 = self.maxpool(x2)# 32x7x7
        x3 = self.conv_3(x3) # 64x7x7

        d3 = self.up_3(x3) # 32x14x14
        d3 = torch.cat((x2, d3), dim=1)  # 32x14x14
        d3 = self.up_conv_3(d3)  # 32x14x14

        d2 = self.up_2(d3) # 16x28x28
        d2 = torch.cat((x1, d2), dim=1)  # 16x28x28
        d2 = self.up_conv_2(d2) # 16x28x28

        d1 = self.conv_out(d2) # 1x28x28
        return d1
