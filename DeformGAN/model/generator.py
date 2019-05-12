import torch
from torch import nn

from DeformGAN.model.modules import GenEncConvLayer, GenDecConvLayer


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.source_enc_conv_1 = GenEncConvLayer(21, 64, stride=1)
        self.source_enc_conv_2 = GenEncConvLayer(64, 128)
        self.source_enc_conv_3 = GenEncConvLayer(128, 256)
        self.source_enc_conv_4 = GenEncConvLayer(256, 512)
        self.source_enc_conv_5 = GenEncConvLayer(512, 512)
        self.source_enc_conv_6 = GenEncConvLayer(512, 512)
        self.source_enc_conv_7 = GenEncConvLayer(512, 512)

        # TODO: This stuff can be cached if rewritten
        self.target_enc_conv_1 = GenEncConvLayer(18, 64, stride=1)
        self.target_enc_conv_2 = GenEncConvLayer(64, 128)
        self.target_enc_conv_3 = GenEncConvLayer(128, 256)
        self.target_enc_conv_4 = GenEncConvLayer(256, 512)
        self.target_enc_conv_5 = GenEncConvLayer(512, 512)
        self.target_enc_conv_6 = GenEncConvLayer(512, 512)
        self.target_enc_conv_7 = GenEncConvLayer(512, 512)

        self.dec_conv_1 = GenDecConvLayer(512*2, 512, dropout=True)
        self.dec_conv_2 = GenDecConvLayer(512*3, 512, dropout=True)
        self.dec_conv_3 = GenDecConvLayer(512*3, 512, dropout=True)
        self.dec_conv_4 = GenDecConvLayer(512*3, 512)
        self.dec_conv_5 = GenDecConvLayer(512*3, 256)
        self.dec_conv_6 = GenDecConvLayer(256*3, 128)
        self.dec_conv_7 = nn.Conv2d(128*3, 3, 3, stride=1)

    def forward(self, app_img, app_hms, pose_hms):
        source_enc_inp = torch.cat([app_img, app_hms], dim=1)
        source_enc_x_1 = self.source_enc_conv_1(source_enc_inp)
        source_enc_x_2 = self.source_enc_conv_2(source_enc_x_1)
        source_enc_x_3 = self.source_enc_conv_3(source_enc_x_2)
        source_enc_x_4 = self.source_enc_conv_4(source_enc_x_3)
        source_enc_x_5 = self.source_enc_conv_5(source_enc_x_4)
        source_enc_x_6 = self.source_enc_conv_6(source_enc_x_5)
        source_enc_x_7 = self.source_enc_conv_7(source_enc_x_6)

        # TODO: This can be cached if rewritten
        target_enc_x_1 = self.target_enc_conv_1(pose_hms)
        target_enc_x_2 = self.target_enc_conv_2(target_enc_x_1)
        target_enc_x_3 = self.target_enc_conv_3(target_enc_x_2)
        target_enc_x_4 = self.target_enc_conv_4(target_enc_x_3)
        target_enc_x_5 = self.target_enc_conv_5(target_enc_x_4)
        target_enc_x_6 = self.target_enc_conv_6(target_enc_x_5)
        target_enc_x_7 = self.target_enc_conv_7(target_enc_x_6)

        x = self.dec_conv_1(source_enc_x_7, target_enc_x_7)
        x = self.dec_conv_2(source_enc_x_6, target_enc_x_6, x)
        x = self.dec_conv_3(source_enc_x_5, target_enc_x_5, x)
        x = self.dec_conv_4(source_enc_x_4, target_enc_x_4, x)
        x = self.dec_conv_5(source_enc_x_3, target_enc_x_3, x)
        x = self.dec_conv_6(source_enc_x_2, target_enc_x_2, x)
        x = self.dec_conv_7(source_enc_x_1, target_enc_x_1, x)



class TargetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = GenEncConvLayer(18, 64, stride=1)
        self.conv_2 = GenEncConvLayer(64, 128)
        self.conv_3 = GenEncConvLayer(128, 256)
        self.conv_4 = GenEncConvLayer(256, 512)
        self.conv_5 = GenEncConvLayer(512, 512)
        self.conv_6 = GenEncConvLayer(512, 512)
        self.conv_7 = GenEncConvLayer(512, 512)

    def forward(self, pose_hms):
        x = self.conv_1(pose_hms)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
        x = self.conv_7(x)
        return x


class SourceDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = GenEncConvLayer(512, 512, dropout=True)
        self.conv_2 = GenEncConvLayer(512, 512, dropout=True)
        self.conv_3 = GenEncConvLayer(512, 512, dropout=True)
        self.conv_4 = GenEncConvLayer(512, 512)
        self.conv_5 = GenEncConvLayer(512, 256)
        self.conv_6 = GenEncConvLayer(256, 128)
        self.conv_7 = nn.Conv2d(128, 3, 3, stride=1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
        x = self.conv_7(x)
        # TODO:Check if this should be sigmoid
        x = nn.Tanh()(x)
        return x


class TargetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = GenEncConvLayer(512, 512, dropout=True)
        self.conv_2 = GenEncConvLayer(512, 512, dropout=True)
        self.conv_3 = GenEncConvLayer(512, 512, dropout=True)
        self.conv_4 = GenEncConvLayer(512, 512)
        self.conv_5 = GenEncConvLayer(512, 256)
        self.conv_6 = GenEncConvLayer(256, 128)
        self.conv_7 = nn.Conv2d(128, 3, 3, stride=1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
        x = self.conv_7(x)
        # TODO:Check if this should be sigmoid
        x = nn.Tanh()(x)
        return x
