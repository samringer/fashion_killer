import torch
from torch import nn
import torch.utils.checkpoint as chk

from app_transfer.model.spectral_norm import SpecNorm
from app_transfer.model.std_attention import AttentionMech


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_0 = DisConvLayer(48, 128)
        self.conv_1 = DisConvLayer(128, 128)
        #self.conv_2 = DisConvLayer(128, 128, stride=1, padding=1)
        # TODO: Update this properly when moving to larger dims
        self.attn = AttentionMech(128)
        self.conv_3 = DisConvLayer(128, 256)
        self.conv_4 = DisConvLayer(256, 512)
        self.conv_5 = DisConvLayer(512, 512)
        self.conv_6 = nn.Conv2d(512, 1, 4)

    def custom(self, module):
        """
        Gradient checkpoint the attention blocks as they are very
        memory intensive.
        """
        def custom_forward(*inp):
            out = module(inp[0])
            return out
        return custom_forward

    def forward(self, app_img, app_pose_img, pose_img, target_img):
        x = torch.cat([app_img, app_pose_img, pose_img, target_img], dim=1)
        x = self.conv_0(x)
        x = self.conv_1(x)
        #x = self.conv_2(x)
        x = chk.checkpoint(self.custom(self.attn), x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
        return x

    def get_features(self, app_img, app_pose_img, pose_img, target_img):
        """
        Extract features to be used for the pyramid heirachy loss.
        """
        x = torch.cat([app_img, app_pose_img, pose_img, target_img], dim=1)
        f_0 = self.conv_0(x)
        f_1 = self.conv_1(f_0)
        #x = self.conv_2(x)
        f_2 = chk.checkpoint(self.custom(self.attn), f_1)
        return [f_0, f_1, f_2]

    def hierachy_loss(self, app_img, app_pose_img, pose_img,
                      target_img, gen_img):
        hierarchy_loss = 0
        gen_feats = self.get_features(app_img, app_pose_img,
                                      pose_img, gen_img)
        with torch.no_grad():
            real_feats = self.get_features(app_img, app_pose_img,
                                           pose_img, target_img)

        for gen_feat, real_feat in zip(gen_feats, real_feats):
            hierarchy_loss += nn.L1Loss()(gen_feat, real_feat)

        return hierarchy_loss


class DisConvLayer(nn.Module):
    def __init__(self, in_c, out_c, stride=2, padding=1):
        super().__init__()
        self.conv = SpecNorm(nn.Conv2d(in_c, out_c, 3, stride=stride,
                                       padding=padding))

    def forward(self, x):
        x = self.conv(x)
        x = nn.ReLU()(x)
        return x
