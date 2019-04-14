import torch
from torch import nn
from v_u_net.model.modules import (EncoderBlock, DecoderBlock,
                                   EncResLayer, DecResLayer)


class UNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_1x1 = nn.Conv2d(3, 64, 1)

        self.enc_block_1 = EncoderBlock(64, 128)
        self.enc_block_2 = EncoderBlock(128, 256)
        self.enc_block_3 = EncoderBlock(256, 256)
        self.enc_block_4 = EncoderBlock(256, 256)
        self.enc_block_5 = EncoderBlock(256, 256)
        self.enc_block_6 = EncoderBlock(256, 256)
        self.enc_block_7 = EncoderBlock(256, 256)
        self.enc_block_8 = EncoderBlock(256, 256)

        self.enc_final_res_1 = EncResLayer(256)
        self.enc_final_conv = nn.Conv2d(256, 256, 1)
        self.enc_final_res_2 = EncResLayer(256)

        self.dec_initial_res_1 = DecResLayer(256*2)
        self.dec_initial_res_2 = DecResLayer(256*2)
        self.dec_custom_conv = nn.Conv2d(512, 256, 3, padding=1)
        self.dec_initial_res_3 = DecResLayer(256*2)

        self.dec_block_1 = DecoderBlock(256, 256)
        self.dec_block_2 = DecoderBlock(256, 256)
        self.dec_block_3 = DecoderBlock(256, 256)
        self.dec_block_4 = DecoderBlock(256, 256)
        self.dec_block_5 = DecoderBlock(256, 256)
        self.dec_block_6 = DecoderBlock(256, 128)
        self.dec_block_7 = DecoderBlock(128, 64)
        self.dec_final_res = DecResLayer(64*2)
        self.dec_final_conv = nn.Conv2d(64*2, 3, 1)

    def forward(self, pose_img, app_vec_1x1, app_vec_2x2):
        x_orig = self.conv_1x1(pose_img)
        x_1a, x_1b = self.enc_block_1(x_orig)
        x_2a, x_2b = self.enc_block_2(x_1b)
        x_3a, x_3b = self.enc_block_3(x_2b)
        x_4a, x_4b = self.enc_block_4(x_3b)
        x_5a, x_5b = self.enc_block_5(x_4b)
        x_6a, x_6b = self.enc_block_6(x_5b)
        x_7a, x_7b = self.enc_block_7(x_6b)
        x_8a, x_8b = self.enc_block_8(x_7b)

        x_9 = self.enc_final_res_1(x_8b)
        x_10 = self.enc_final_conv(x_9)
        x_11 = torch.cat((x_9, x_10), dim=1)
        pose_vec = self.dec_initial_res_1(x_11)

        pose_mu_1x1 = x_9
        pose_mu_2x2 = x_8a

        # This block is very custom to match the paper exactly
        x = torch.cat((pose_vec, app_vec_1x1), dim=1)
        x = self.dec_initial_res_2(x)
        x = torch.cat((x, x_8b), dim=1)
        x = nn.functional.interpolate(x, scale_factor=2)
        x = self.dec_custom_conv(x)
        x = torch.cat((x, x_8a), dim=1)
        x = self.dec_initial_res_3(x)

        x = self.dec_block_1(x, app_vec_2x2, x_7b)
        x = self.dec_block_2(x, x_7a, x_6b)
        x = self.dec_block_3(x, x_6a, x_5b)
        x = self.dec_block_4(x, x_5a, x_4b)
        x = self.dec_block_5(x, x_4a, x_3b)
        x = self.dec_block_6(x, x_3a, x_2b)
        x = self.dec_block_7(x, x_2a, x_1b)

        x = torch.cat((x, x_1a), dim=1) 
        x = self.dec_final_res(x)
        x = torch.cat((x, x_orig), dim=1)
        x = self.dec_final_conv(x)
        return x, pose_mu_1x1, pose_mu_2x2

    def sample_from_gaussian(self, mu):
        # std is assumed to be constant at 1
        eps = torch.randn_like(mu)
        sample = eps.add_(mu)
        return sample
