import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as chk

from GAN.model.spectral_norm import SpecNorm
from GAN.model.attention import AttentionMech
from GAN.model.modules import (EncoderBlock, DecoderBlock,
                               EncResLayer, DecResLayer)


class UNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_1x1 = SpecNorm(nn.Conv2d(6, 64, 1))

        self.enc_block_1 = EncoderBlock(64, 128)
        self.enc_block_2 = EncoderBlock(128, 256)
        self.enc_block_3 = EncoderBlock(256, 256)

        self.attn_1 = AttentionMech(256)

        self.enc_block_4 = EncoderBlock(256, 256)
        self.enc_block_5 = EncoderBlock(256, 256)
        self.enc_block_6 = EncoderBlock(256, 256)
        self.enc_block_7 = EncoderBlock(256, 256)
        #self.enc_block_8 = EncoderBlock(256, 256)

        self.enc_final = EncResLayer(256)

        self.dec_first = DecResLayer(512)

        #self.dec_block_1 = DecoderBlock(256, 256)
        self.dec_block_2 = DecoderBlock(256, 256)
        self.dec_block_3 = DecoderBlock(256, 256)
        self.dec_block_4 = DecoderBlock(256, 256)

        self.attn_2 = AttentionMech(256)

        self.dec_block_5 = DecoderBlock(256, 256)
        self.dec_block_6 = DecoderBlock(256, 128)
        self.dec_block_7 = DecoderBlock(128, 64)
        self.dec_final_res = DecResLayer(64*2)
        self.dec_final_conv = SpecNorm(nn.Conv2d(64*2, 3, 1))

    def _chk_block(self, block):
        """ Used for gradient checkpointing."""
        def _custom_forward(*inputs):
            outputs = block(inputs[0])
            return outputs
        return _custom_forward

    def forward(self, app_img, pose_img):
        x_in = torch.cat([app_img, pose_img], dim=1)
        print('hi', x_in.shape)
        x_0 = self.conv_1x1(x_in)
        print('0', x_0.shape)
        x_1a, x_1b = chk(self._chk_block(self.enc_block_1), x_0)
        x_2a, x_2b = self.enc_block_2(x_1b)
        x_3a, x_3b = self.enc_block_3(x_2b)
        x_3b = self.attn_1(x_3b)
        x_4a, x_4b = self.enc_block_4(x_3b)
        x_5a, x_5b = self.enc_block_5(x_4b)
        print('1', x_5a.shape)
        x_6a, x_6b = self.enc_block_6(x_5b)
        x_7a, x_7b = self.enc_block_7(x_6b)
        #x_8a, x_8b = self.enc_block_8(x_7b)

        x_bottom = self.enc_final(x_7b)
        print('2', x_bottom.shape)

        #x = torch.cat([x_bottom, x_8b], dim=1)
        x = torch.cat([x_bottom, x_7b], dim=1)
        x = nn.functional.interpolate(x, scale_factor=2)
        x = self.dec_first(x)

        #x = self.dec_block_1(x, x_8a, x_7b)
        x = self.dec_block_2(x, x_7a, x_6b)
        x = self.dec_block_3(x, x_6a, x_5b)
        x = self.dec_block_4(x, x_5a, x_4b)
        print('3', x.shape)

        x = self.attn_2(x)

        x = self.dec_block_5(x, x_4a, x_3b)
        x = self.dec_block_6(x, x_3a, x_2b)
        x = self.dec_block_7(x, x_2a, x_1b)

        x = torch.cat([x, x_1a], dim=1)
        x = self.dec_final_res(x)
        print('4', x.shape)
        x = torch.cat([x, x_0], dim=1)
        gen_img = self.dec_final_conv(x)
        gen_img = nn.Sigmoid()(gen_img)
        print('5', gen_img.shape)
        return gen_img
