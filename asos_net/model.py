import torch
from torch import nn


class AsosNet(nn.Module):


    def __init__(self):
        super().__init__()
        in_c = 6  # Pose tensor (RGB) catted with app tensor (RGB)
        self.block_1 = AsosBlock(in_c)
        self.block_2 = AsosBlock(in_c+3)
        self.block_3 = AsosBlock(in_c+3)
        self.block_4 = AsosBlock(in_c+3)

    def forward(self, app_tensor, pose_tensor):

        # TODO: Im sure lots of model space can be saved by scaling
        # down to a smaller resolution or something
        inp = torch.cat(app_tensor, pose_tensor)
        out_1 = self.block_1(inp)

        x = torch.cat(out_1, inp)
        out_2 = self.block_2(x)

        x = torch.cat(out_2, inp)
        out_3 = self.block_3(x)

        x = torch.cat(out_3, inp)
        out_4 = self.block_4(x)

        return out_1, out_2, out_3, out_4


class AsosBlock(nn.Module):

    def __init__(self, in_c):
        super().__init__()
        self.conv_1x1b = ConvLayer(in_c, 32, kernel_size=1, padding=0)
        self.conv_block_1 = ConvBlock(32)
        self.conv_block_2 = ConvBlock(32)
        self.conv_block_3 = ConvBlock(32)
        self.conv_block_4 = ConvBlock(32)
        self.conv_block_5 = ConvBlock(32)
        self.conv_1x1b = ConvLayer(32, 32, kernel_size=1, padding=0)
        self.conv_1x1c = ConvLayer(32, 3, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv_1x1a(x)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.conv_block_5(x)
        x = self.conv_1x1b(x)
        x = self.conv_1x1c(x)
        # TODO: Check if this helps or not
        x = nn.Sigmoid()(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.conv_1 = ConvLayer(in_c, in_c)
        self.conv_2 = ConvLayer(in_c, in_c)
        self.conv_3 = ConvLayer(in_c, in_c)

    def forward(self, x_0):
        x_1 = self.conv_1(x_0)
        x_2 = self.conv_2(x_1)
        x_3 = self.conv_3(x_2)
        out = x_0 + x_1 + x_2 + x_3
        return out


class ConvLayer(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, padding=padding,
                              bias=False)
        self.l_relu = nn.LeakyReLU(negative_slope=0.2)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.l_relu(x)
        x = self.bn(x)
        return x
