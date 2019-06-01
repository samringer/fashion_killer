import torch
from torch import nn

from GAN.model.spectral_norm import SpecNorm


class EncoderBlock(nn.Module):
    """
    We need to return both x_1 and x_2 as the Unet uses
    both for it's long skip connections
    """

    def __init__(self, in_c, out_c):
        super().__init__()
        self.res_block = EncResLayer(in_c)
        # Performs downsampling
        self.conv = SpecNorm(nn.Conv2d(in_c, out_c, 3, stride=2, padding=1))

    def forward(self, x):
        x_1 = self.res_block(x)
        x_2 = self.conv(x_1)
        return x_1, x_2


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.res_layer = DecResLayer(in_c*2)
        self.conv = SpecNorm(nn.Conv2d(in_c*2, out_c, 3, padding=1))

    def forward(self, x, skip_con_1, skip_con_2):
        x = torch.cat((x, skip_con_1), dim=1)
        x = self.res_layer(x)
        x = torch.cat((x, skip_con_2), dim=1)
        x = nn.functional.interpolate(x, scale_factor=2)
        x = self.conv(x)
        return x


class EncResLayer(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.conv_1 = SpecNorm(nn.Conv2d(in_c, in_c, kernel_size=3,
                                         padding=1, bias=False))
        self.conv_2 = SpecNorm(nn.Conv2d(in_c, in_c, kernel_size=3,
                                         padding=1, bias=False))
        self.batch_norm_1 = nn.BatchNorm2d(in_c)
        self.batch_norm_2 = nn.BatchNorm2d(in_c)
        self.l_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        f_x = self.conv_1(x)
        f_x = self.batch_norm_1(f_x)
        f_x = self.l_relu(f_x)
        f_x = self.conv_2(f_x)
        f_x = self.batch_norm_2(f_x)
        x = f_x + x
        x = self.l_relu(x)
        return x


class DecResLayer(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.conv_1 = SpecNorm(nn.Conv2d(in_c, in_c, 3, padding=1,
                                         bias=False))
        self.conv_2 = SpecNorm(nn.Conv2d(in_c, in_c//2, 3, padding=1,
                                         bias=False))
        self.proj = SpecNorm(nn.Conv2d(in_c, in_c//2, 1))
        self.l_relu = nn.LeakyReLU(negative_slope=0.2)

        self.batch_norm_1 = nn.BatchNorm2d(in_c)
        self.batch_norm_2 = nn.BatchNorm2d(in_c//2)

    def forward(self, x):
        f_x = self.conv_1(x)
        f_x = self.batch_norm_1(f_x)
        f_x = self.l_relu(f_x)
        f_x = self.conv_2(f_x)
        f_x = self.batch_norm_2(f_x)
        f_x = f_x + self.proj(x)
        f_x = self.l_relu(f_x)
        return f_x