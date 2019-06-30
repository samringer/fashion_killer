import torch
import torch.utils.checkpoint as chk
from torch import nn

from absl import logging


class GenEncConvLayer(nn.Module):
    def __init__(self, in_c, out_c, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1)
        self.instance_norm = nn.InstanceNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.instance_norm(x)
        x = nn.ReLU()(x)
        return x


class GenDecAttnBlock(nn.Module):
    def __init__(self, in_c, prev_in_c, out_c, dropout=False,
                 downsample_fac=1):
        super().__init__()
        self.attn_mech = AttnMech(in_c, downsample_fac)
        self.conv = GenDecConvLayer(in_c, prev_in_c, out_c,
                                    dropout=dropout)

    def custom(self, module):
        """
        Gradient checkpoint the attention blocks as they are very
        memory intensive.
        """
        def custom_forward(*inp):
            out = module(inp[0], inp[1])
            return out
        return custom_forward

    def forward(self, source_enc_f, target_enc_f, prev_inp):
        x = chk.checkpoint(self.custom(self.attn_mech), source_enc_f, target_enc_f)
        return self.conv(x, prev_inp)


class GenDecConvBlock(nn.Module):
    def __init__(self, in_c, prev_in_c, out_c, dropout=False):
        super().__init__()
        self.conv = GenDecConvLayer(in_c*2, prev_in_c, out_c,
                                    dropout=dropout)

    def forward(self, source_enc_f, target_enc_f, prev_inp):
        x = torch.cat([source_enc_f, target_enc_f], dim=1)
        return self.conv(x, prev_inp)


class GenDecConvLayer(nn.Module):
    def __init__(self, in_c, prev_in_c, out_c, dropout=False):
        super().__init__()
        self.conv = nn.Conv2d(in_c+prev_in_c, out_c, 3, padding=1)
        self.instance_norm = nn.InstanceNorm2d(out_c)
        self.dropout = dropout

    def forward(self, x, prev_inp):
        prev_inp = nn.functional.interpolate(prev_inp, scale_factor=2)
        x = torch.cat([x, prev_inp], dim=1)
        if self.dropout:
            x = nn.Dropout()(x)
        x = self.conv(x)
        x = self.instance_norm(x)
        x = nn.ReLU()(x)
        return x


class AttnMech(nn.Module):
    def __init__(self, in_c, downsample_fac=1):
        """
        Can optionally perform downsampling so we have a method similar
        to sparse attention.
        """
        super().__init__()
        self.attn_size = in_c//2
        self.in_c = in_c
        self.k_conv = nn.Conv2d(in_c, self.attn_size,
                                kernel_size=downsample_fac,
                                stride=downsample_fac)
        self.q_conv = nn.Conv2d(in_c, self.attn_size,
                                kernel_size=downsample_fac,
                                stride=downsample_fac)
        self.v_conv = nn.Conv2d(in_c, in_c,
                                kernel_size=downsample_fac,
                                stride=downsample_fac)
        self.upsample = nn.Upsample(scale_factor=downsample_fac)
        self.gamma = nn.Parameter(torch.Tensor([0.]))

    def forward(self, source_enc_f, target_enc_f):
        query = self.q_conv(target_enc_f)
        key = self.k_conv(source_enc_f)
        value = self.v_conv(source_enc_f)

        _, _, w, h = query.shape

        query = query.view(-1, self.attn_size, w*h).transpose(1, 2)
        key = key.view(-1, self.attn_size, w*h)
        value = value.permute(0, 2, 3, 1).contiguous().view(-1, w*h, self.in_c)

        attn = query@key
        attn = nn.Softmax(dim=1)(attn)
        value = attn@value
        value = value.view(-1, w, h, self.in_c).permute(0, 3, 1, 2)
        value = self.upsample(value)
        return target_enc_f + self.gamma * value
