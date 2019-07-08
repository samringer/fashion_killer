import torch
from torch import nn
import torch.utils.checkpoint as chk

from DeformGAN.model.spectral_norm import SpecNorm
from DeformGAN.model.std_attention import AttentionMech


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = DisConvLayer(12, 128)
        self.conv_2 = DisConvLayer(128, 128)
        self.conv_3 = DisConvLayer(128, 128, stride=1, padding=1)
        self.conv_4 = DisConvLayer(128, 256)
        self.conv_5 = DisConvLayer(256, 512)
        self.attn = AttentionMech(512)
        self.conv_6 = DisConvLayer(512, 512)
        self.conv_7 = nn.Conv2d(512, 1, 3)

    def custom(self, module):
        """
        Gradient checkpoint the attention blocks as they are very
        memory intensive.
        """
        def custom_forward(*inp):
            out = module(inp[0])
            return out
        return custom_forward

    def forward(self, app_img, app_pose_img, target_img, pose_img):
        x = torch.cat([app_img, app_pose_img, target_img, pose_img], dim=1)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = chk.checkpoint(self.custom(self.attn), x)
        x = self.conv_6(x)
        x = self.conv_7(x)
        return x


class DisConvLayer(nn.Module):
    def __init__(self, in_c, out_c, stride=2, padding=0):
        super().__init__()
        self.conv = SpecNorm(nn.Conv2d(in_c, out_c, 3, stride=stride,
                                       padding=padding))

    def forward(self, x):
        x = self.conv(x)
        x = nn.ReLU()(x)
        return x
