import torch
from torch import nn
from GAN.model.spectral_norm import SpecNorm
from GAN.model.attention import AttentionMech


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.from_RGB = ConvLayer(9, 64, kernel_size=1, padding=0)
        self.block_1 = ConvBlock(64, 128, pool=False)
        self.block_2 = ConvBlock(128, 256, pool=False)
        self.block_3 = ConvBlock(256, 512)
        self.block_4 = ConvBlock(512, 512)
        self.block_5 = ConvBlock(512, 512)
        self.block_6 = ConvBlock(512, 512)
        #self.block_7 = ConvBlock(512, 512)

        self.attn_1 = AttentionMech(512)
        self.attn_2 = AttentionMech(512)

        self.fc = SpecNorm(nn.Linear(512*4*4, 1))

    def forward(self, app_img, pose_img, gen_img):
        x = torch.cat([app_img, pose_img, gen_img], dim=1)
        x = self.from_RGB(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.attn_1(x)
        x = self.block_4(x)
        x = self.attn_2(x)
        x = self.block_5(x)
        x = self.block_6(x)
        #x = self.block_7(x)
        x = x.view(-1, 4*4*512)
        return self.fc(x)


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, pool=True):
        super().__init__()
        self.conv_1 = ConvLayer(in_c, in_c)
        self.conv_2 = ConvLayer(in_c, out_c)
        self.pool = pool
        self.ave_pool = nn.AvgPool2d(kernel_size=2)

        if in_c != out_c:
            self.res_conv = SpecNorm(nn.Conv2d(in_c, out_c,
                                               kernel_size=1, stride=1))

    def forward(self, x):
        f_x = self.conv_1(x)
        f_x = self.conv_2(f_x)
        if hasattr(self, 'res_conv'):
            x = f_x + self.res_conv(x)
        else:
            x = f_x + x
        if self.pool:
            x = self.ave_pool(x)
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1):
        super().__init__()
        self.conv = SpecNorm(nn.Conv2d(in_c, out_c, kernel_size,
                                       padding=padding))
        self.l_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.conv(x)
        return self.l_relu(x)


class FinalBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = ConvLayer(512, 512)
        self.conv_2 = ConvLayer(512, 512, 4, padding=0)
        self.linear = SpecNorm(nn.Linear(512, 1, bias=False))

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x.squeeze_().squeeze_()
        x = self.linear(x)
        return x
