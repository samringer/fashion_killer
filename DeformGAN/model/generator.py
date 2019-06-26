import torch
from torch import nn

from DeformGAN.model.modules import (GenEncConvLayer,
                                     GenDecConvBlock,
                                     GenDecAttnBlock,
                                     AttnMech)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.source_enc_conv_1 = GenEncConvLayer(6, 64, stride=1)
        self.source_enc_conv_2 = GenEncConvLayer(64, 128)
        self.source_enc_conv_3 = GenEncConvLayer(128, 256)
        self.source_enc_conv_4 = GenEncConvLayer(256, 512)
        self.source_enc_conv_5 = GenEncConvLayer(512, 512)
        self.source_enc_conv_6 = GenEncConvLayer(512, 512)
        #self.source_enc_conv_7 = GenEncConvLayer(512, 512)
        #self.source_enc_conv_8 = GenEncConvLayer(512, 512)

        # TODO: This stuff can be cached if rewritten
        self.target_enc_conv_1 = GenEncConvLayer(3, 64, stride=1)
        self.target_enc_conv_2 = GenEncConvLayer(64, 128)
        self.target_enc_conv_3 = GenEncConvLayer(128, 256)
        self.target_enc_conv_4 = GenEncConvLayer(256, 512)
        self.target_enc_conv_5 = GenEncConvLayer(512, 512)
        self.target_enc_conv_6 = GenEncConvLayer(512, 512)
        #self.target_enc_conv_7 = GenEncConvLayer(512, 512)
        #self.target_enc_conv_8 = GenEncConvLayer(512, 512)

        self.dec_conv_1 = GenDecAttnBlock(512, 1024, 512, dropout=True)
        #self.dec_conv_2 = GenDecAttnBlock(512, 512, 512, dropout=True)
        #self.dec_conv_3 = GenDecConvLayer(512*3, 512, dropout=True)
        self.dec_conv_4 = GenDecAttnBlock(512, 512, 512)
        self.dec_conv_5 = GenDecAttnBlock(256, 512, 256)
        self.dec_conv_6 = GenDecConvBlock(128, 256, 128)
        self.dec_conv_7 = GenDecConvBlock(64, 128, 64)
        self.dec_conv_8 = nn.Conv2d(64, 3, 3, stride=1, padding=1)

    def forward(self, app_img, app_pose_img, pose_img):
        source_enc_inp = torch.cat([app_img, app_pose_img], dim=1)
        source_enc_x_1 = self.source_enc_conv_1(source_enc_inp)
        source_enc_x_2 = self.source_enc_conv_2(source_enc_x_1)
        source_enc_x_3 = self.source_enc_conv_3(source_enc_x_2)
        source_enc_x_4 = self.source_enc_conv_4(source_enc_x_3)
        source_enc_x_5 = self.source_enc_conv_5(source_enc_x_4)
        source_enc_x_6 = self.source_enc_conv_6(source_enc_x_5)
        #source_enc_x_7 = self.source_enc_conv_7(source_enc_x_6)
        #source_enc_x_8 = self.source_enc_conv_8(source_enc_x_7)

        # TODO: This can be cached if rewritten
        target_enc_x_1 = self.target_enc_conv_1(pose_img)
        target_enc_x_2 = self.target_enc_conv_2(target_enc_x_1)
        target_enc_x_3 = self.target_enc_conv_3(target_enc_x_2)
        target_enc_x_4 = self.target_enc_conv_4(target_enc_x_3)
        target_enc_x_5 = self.target_enc_conv_5(target_enc_x_4)
        target_enc_x_6 = self.target_enc_conv_6(target_enc_x_5)
        #target_enc_x_7 = self.target_enc_conv_7(target_enc_x_6)
        #target_enc_x_8 = self.target_enc_conv_8(target_enc_x_7)

        x = torch.cat([source_enc_x_6, target_enc_x_6], dim=1)
        x = self.dec_conv_1(source_enc_x_5, target_enc_x_5, x)
        #x = torch.cat([source_enc_x_8, target_enc_x_8], dim=1)
        #x = self.dec_conv_1(source_enc_x_7, target_enc_x_7, x)
        #x = self.dec_conv_2(source_enc_x_5, target_enc_x_5, x)
        #x = self.dec_conv_3(source_enc_x_5, target_enc_x_5, x)
        x = self.dec_conv_4(source_enc_x_4, target_enc_x_4, x)
        x = self.dec_conv_5(source_enc_x_3, target_enc_x_3, x)
        x = self.dec_conv_6(source_enc_x_2, target_enc_x_2, x)
        x = self.dec_conv_7(source_enc_x_1, target_enc_x_1, x)
        x = self.dec_conv_8(x)
        x = nn.Sigmoid()(x)
        return x
