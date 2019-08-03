import torch
from torch import nn

from app_transfer.model.modules import (GenDecConvBlock,
                                        GenDecAttnBlock,
                                        ConvBlock)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.app_enc_conv_0 = ConvBlock(3, 64)
        self.app_enc_conv_1 = ConvBlock(64, 128, stride=2)
        self.app_enc_conv_2 = ConvBlock(128, 256, stride=2)
        self.app_enc_conv_3 = ConvBlock(256, 512, stride=2)
        self.app_enc_conv_4 = ConvBlock(512, 512, stride=2)
        self.app_enc_conv_5 = ConvBlock(512, 512, stride=2)
        self.app_enc_conv_6 = ConvBlock(512, 512, stride=2)
        #self.app_enc_conv_7 = ConvBlock(512, 512, stride=2)

        self.app_pose_enc_conv_0 = ConvBlock(21, 64)
        self.app_pose_enc_conv_1 = ConvBlock(64, 128, stride=2)
        self.app_pose_enc_conv_2 = ConvBlock(128, 256, stride=2)
        self.app_pose_enc_conv_3 = ConvBlock(256, 512, stride=2)
        self.app_pose_enc_conv_4 = ConvBlock(512, 512, stride=2)
        self.app_pose_enc_conv_5 = ConvBlock(512, 512, stride=2)
        self.app_pose_enc_conv_6 = ConvBlock(512, 512, stride=2)
        #self.app_pose_enc_conv_7 = ConvBlock(512, 512, stride=2)

        self.pose_enc_conv_0 = ConvBlock(21, 64)
        self.pose_enc_conv_1 = ConvBlock(64, 128, stride=2)
        self.pose_enc_conv_2 = ConvBlock(128, 256, stride=2)
        self.pose_enc_conv_3 = ConvBlock(256, 512, stride=2)
        self.pose_enc_conv_4 = ConvBlock(512, 512, stride=2)
        self.pose_enc_conv_5 = ConvBlock(512, 512, stride=2)
        self.pose_enc_conv_6 = ConvBlock(512, 512, stride=2)
        #self.pose_enc_conv_7 = ConvBlock(512, 512, stride=2)

        self.dec_conv_0 = GenDecConvBlock(1536, 1536, 512)
        # comment below out for 64x64
        self.dec_conv_1 = GenDecConvBlock(1536, 512, 512)

        # comment below out for 128x128
        #self.dec_conv_2 = GenDecConvBlock(1536, 512, 512)

        self.dec_conv_3 = GenDecAttnBlock(512, 512, 512)
        self.dec_conv_4 = GenDecAttnBlock(256, 512, 256, num_heads=2)

        self.dec_conv_5 = ConvBlock(256, 128, upsample=True)
        self.dec_conv_6 = ConvBlock(128, 64, upsample=True)
        self.dec_conv_7 = ConvBlock(64, 64)
        self.dec_conv_8 = nn.Conv2d(64, 3, 1)

    def forward(self, app_img, app_pose_img, pose_img):
        app_enc_0 = self.app_enc_conv_0(app_img)
        app_enc_1 = self.app_enc_conv_1(app_enc_0)
        app_enc_2 = self.app_enc_conv_2(app_enc_1)
        app_enc_3 = self.app_enc_conv_3(app_enc_2)
        app_enc_4 = self.app_enc_conv_4(app_enc_3)
        app_enc_5 = self.app_enc_conv_5(app_enc_4)
        app_enc_6 = self.app_enc_conv_6(app_enc_5)
        #app_enc_7 = self.app_enc_conv_7(app_enc_6)

        app_pose_enc_0 = self.app_pose_enc_conv_0(app_pose_img)
        app_pose_enc_1 = self.app_pose_enc_conv_1(app_pose_enc_0)
        app_pose_enc_2 = self.app_pose_enc_conv_2(app_pose_enc_1)
        app_pose_enc_3 = self.app_pose_enc_conv_3(app_pose_enc_2)
        app_pose_enc_4 = self.app_pose_enc_conv_4(app_pose_enc_3)
        app_pose_enc_5 = self.app_pose_enc_conv_5(app_pose_enc_4)
        app_pose_enc_6 = self.app_pose_enc_conv_6(app_pose_enc_5)
        #app_pose_enc_7 = self.app_pose_enc_conv_7(app_pose_enc_6)

        pose_enc_0 = self.pose_enc_conv_0(pose_img)
        pose_enc_1 = self.pose_enc_conv_1(pose_enc_0)
        pose_enc_2 = self.pose_enc_conv_2(pose_enc_1)
        pose_enc_3 = self.pose_enc_conv_3(pose_enc_2)
        pose_enc_4 = self.pose_enc_conv_4(pose_enc_3)
        pose_enc_5 = self.pose_enc_conv_5(pose_enc_4)
        pose_enc_6 = self.pose_enc_conv_6(pose_enc_5)
        #pose_enc_7 = self.pose_enc_conv_7(pose_enc_6)

        # 64x64
        #x = torch.cat([app_enc_5, app_pose_enc_5, pose_enc_5], dim=1)
        #x = self.dec_conv_0(app_enc_4, app_pose_enc_4, pose_enc_4, x)

        # 128x128
        x = torch.cat([app_enc_6, app_pose_enc_6, pose_enc_6], dim=1)
        x = self.dec_conv_0(app_enc_5, app_pose_enc_5, pose_enc_5, x)
        x = self.dec_conv_1(app_enc_4, app_pose_enc_4, pose_enc_4, x)

        # 256x256
        #x = torch.cat([app_enc_7, app_pose_enc_7, pose_enc_7], dim=1)
        #x = self.dec_conv_0(app_enc_6, app_pose_enc_6, pose_enc_6, x)
        #x = self.dec_conv_1(app_enc_5, app_pose_enc_5, pose_enc_5, x)
        #x = self.dec_conv_2(app_enc_4, app_pose_enc_4, pose_enc_4, x)

        x = self.dec_conv_3(app_enc_3, app_pose_enc_3, pose_enc_3, x)
        x = self.dec_conv_4(app_enc_2, app_pose_enc_2, pose_enc_2, x)
        x = self.dec_conv_5(x)
        x = self.dec_conv_6(x)
        x = self.dec_conv_7(x)
        x = self.dec_conv_8(x)
        x = nn.Sigmoid()(x)
        return x
