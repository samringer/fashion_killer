import torch
from torch import nn

from DeformGAN.model.modules import DisConvLayer


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = DisConvLayer(42, 64)
        self.conv_2 = DisConvLayer(64, 128)
        self.conv_3 = DisConvLayer(128, 256)
        self.conv_4 = DisConvLayer(256, 512)
        #self.conv_5 = DisConvLayer(512, 512)
        #self.conv_6 = DisConvLayer(512, 512)
        self.conv_7 = nn.Conv2d(512, 1, 3)

    def forward(self, app_img, pose_img, app_hms, pose_hms):
        x = torch.cat([app_img, pose_img, app_hms, pose_hms], dim=1)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        #x = self.conv_5(x)
        #x = self.conv_6(x)
        x = self.conv_7(x)
        x = nn.Sigmoid()(x)
        return x
