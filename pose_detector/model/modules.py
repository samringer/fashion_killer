import torch
from torch import nn

import pose_detector.hyperparams as hp


class Limb_Block(nn.Module):

    def __init__(self, in_c):
        super().__init__()
        self.conv_block_1 = Conv_Block(in_c)
        self.conv_block_2 = Conv_Block(in_c)
        self.conv_block_3 = Conv_Block(in_c)
        self.conv_block_4 = Conv_Block(in_c)
        self.conv_block_5 = Conv_Block(in_c)
        self.conv_1x1a = Conv_Layer(in_c, in_c, kernel_size=1, padding=0)
        self.conv_1x1b = Conv_Layer(in_c, hp.num_limbs*2,
                                    kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.conv_block_5(x)
        x = self.conv_1x1a(x)
        x = self.conv_1x1b(x)
        return x

class Joint_Block(nn.Module):

    def __init__(self, in_c):
        super().__init__()
        self.conv_block_1 = Conv_Block(in_c)
        self.conv_block_2 = Conv_Block(in_c)
        self.conv_block_3 = Conv_Block(in_c)
        self.conv_block_4 = Conv_Block(in_c)
        self.conv_block_5 = Conv_Block(in_c)
        self.conv_1x1a = Conv_Layer(in_c, in_c, kernel_size=1, padding=0)
        self.conv_1x1b = Conv_Layer(in_c, hp.num_joints, kernel_size=1, padding=0)

    def forward(self, x_0):
        x = self.conv_block_1(x_0)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.conv_block_5(x)
        x = self.conv_1x1a(x)
        x = self.conv_1x1b(x)
        heat_maps = nn.Sigmoid()(x)
        return heat_maps


class Conv_Block(nn.Module):

    def __init__(self, in_c):
        super().__init__()
        self.conv_1 = Conv_Layer(in_c, in_c)
        self.conv_2 = Conv_Layer(in_c, in_c)
        self.conv_3 = Conv_Layer(in_c, in_c)

    def forward(self, x_0):
        x_1 = self.conv_1(x_0)
        x_2 = self.conv_2(x_1)
        x_3 = self.conv_3(x_2)
        out = x_0 + x_1 + x_2 + x_3
        return out


class Conv_Layer(nn.Module):

    def __init__(self, in_c, out_c, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, padding=padding)
        self.l_relu = nn.LeakyReLU(negative_slope=hp.leakiness)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.l_relu(x)
        x = self.bn(x)
        return x
