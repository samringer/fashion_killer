import torch
from torch import nn
from Fashion_Killer.Model.Modules import Encoder_Block, Enc_Res_Layer

class Appearance_Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_1x1 = nn.Conv2d(3*(1+1+7), 64, 1)
        self.block_1 = Encoder_Block(64, 128)
        self.block_2 = Encoder_Block(128, 256)
        self.block_3 = Encoder_Block(256, 256)
        self.block_4 = Encoder_Block(256, 256)
        self.block_5 = Encoder_Block(256, 256)
        self.block_6 = Encoder_Block(256, 256)
        self.block_7 = Encoder_Block(256, 256)
        self.block_8 = Encoder_Block(256, 256)
        self.enc_final_res = Enc_Res_Layer(256)

    def forward(self, orig_img, pose_img, localised_joints):
        x = torch.cat((orig_img, pose_img, localised_joints), dim=1)
        x = self.conv_1x1(x)
        _, x = self.block_1(x)
        _, x = self.block_2(x)
        _, x = self.block_3(x)
        _, x = self.block_4(x)
        _, x = self.block_5(x)
        _, x = self.block_6(x)
        _, x = self.block_7(x)

        app_mu_2x2, x = self.block_8(x)
        app_mu_1x1 = self.enc_final_res(x)

        app_vec_1x1 = self.sample_from_gaussian(app_mu_1x1)
        app_vec_2x2 = self.sample_from_gaussian(app_mu_2x2)

        return app_vec_1x1, app_vec_2x2, app_mu_1x1, app_vec_2x2

    def sample_from_gaussian(self, mu):
        # std is assumed to be constant at 1
        eps = torch.randn_like(mu)
        sample = eps.add_(mu)
        return sample
