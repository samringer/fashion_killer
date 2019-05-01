import torch
from torch import nn
from GAN.model.conditional_batch_norm import ConditionalBatchNorm2d
from GAN.model.spectral_norm import SpecNorm
from GAN.model.attention import AttentionMech


class Generator(nn.Module):

    def __init__(self):
        super().__init__()
        self.block_1 = InitialConvBlock()
        self.block_2 = UpAndConvBlock(512, 512)
        self.block_3 = UpAndConvBlock(512)
        self.block_4 = UpAndConvBlock(256)
        self.block_5 = UpAndConvBlock(128, upscale=False)
        self.block_6 = UpAndConvBlock(64, upscale=False)

        self.attn_1 = AttentionMech(512)
        self.attn_2 = AttentionMech(256)

        self.toRGB = toRGB(32)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.attn_1(x)
        x = self.block_3(x)
        x = self.attn_2(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.block_6(x)
        x = self.toRGB(x)
        return x


class InitialConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = ConvLayer(512, 512, 4, padding=3)
        self.conv_2 = ConvLayer(512, 512)

    def forward(self, latent_vector):
        x = self.conv_1(latent_vector)
        x = self.conv_2(x)
        return x


class UpAndConvBlock(nn.Module):
    def __init__(self, in_c, out_c=None, upscale=True):
        super().__init__()
        self.upscale = upscale
        if not out_c:
            out_c = int(in_c/2)
        self.conv_1 = ConvLayer(in_c, out_c)
        self.conv_2 = ConvLayer(out_c, out_c)

        if in_c != out_c:
            self.res_conv = SpecNorm(nn.Conv2d(in_c, out_c, 1))

    def forward(self, x):
        if self.upscale:
            x = nn.functional.interpolate(x, scale_factor=2)
        f_x = self.conv_1(x)
        f_x = self.conv_2(f_x)
        if hasattr(self, 'res_conv'):
            return f_x + self.res_conv(x)
        else:
            return f_x + x


class toRGB(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.final_conv = SpecNorm(nn.Conv2d(in_c, 3, 1, padding=0))

    def forward(self, x):
        x = self.final_conv(x)
        x = nn.Sigmoid()(x)
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3,
                 padding=1):
        super().__init__()
        self.conv = SpecNorm(nn.Conv2d(in_c, out_c,
                                       kernel_size=kernel_size,
                                       padding=padding,
                                       bias=False))
        self.batch_norm = nn.BatchNorm2d(out_c)
        self.l_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        return self.l_relu(x)
