import torch
import torch.utils.checkpoint as chk
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.conv = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1)
        self.instance_norm = nn.InstanceNorm2d(out_c)

    def forward(self, x):
        if self.upsample:
            x = nn.Upsample(scale_factor=2)(x)
        x = self.conv(x)
        x = self.instance_norm(x)
        x = nn.ReLU()(x)
        return x


class GenDecAttnBlock(nn.Module):
    def __init__(self, in_c, prev_in_c, out_c, downsample_fac=1):
        super().__init__()
        self.attn_mech = AttnMech(in_c, downsample_fac)
        self.conv = ConvBlock(in_c+prev_in_c, out_c)

    def custom(self, module):
        """
        Gradient checkpoint the attention blocks as they are very
        memory intensive.
        """
        def custom_forward(*inp):
            out = module(inp[0], inp[1], inp[2])
            return out
        return custom_forward

    def forward(self, app_enc, app_pose_enc, pose_enc, prev_inp):
        x = chk.checkpoint(self.custom(self.attn_mech), app_enc,
                           app_pose_enc, pose_enc)
        prev_inp = nn.Upsample(scale_factor=2)(prev_inp)
        x = torch.cat([x, prev_inp], dim=1)
        return self.conv(x)


class GenDecConvBlock(nn.Module):
    def __init__(self, in_c, prev_in_c, out_c):
        super().__init__()
        self.conv = ConvBlock(in_c+prev_in_c, out_c)

    def forward(self, x, y, z, prev_inp):
        x = torch.cat([x, y, z], dim=1)
        prev_inp = nn.functional.interpolate(prev_inp, scale_factor=2)
        x = torch.cat([x, prev_inp], dim=1)
        x = self.conv(x)
        return x


class AttnMech(nn.Module):
    def __init__(self, in_c, downsample_fac=1):
        """
        So far only the hacky option of 2 or for heads is supported.
        Not attn_c is always taken to be in_c//4, regardless of the num
        of heads. This is to make things fit in memory for two headed situtaion.
        """
        super().__init__()
        self.attn_head_1 = SingleAttnHead(in_c, in_c//4, downsample_fac)
        self.attn_head_2 = SingleAttnHead(in_c, in_c//4, downsample_fac)
        self.attn_head_3 = SingleAttnHead(in_c, in_c//4, downsample_fac)
        self.attn_head_4 = SingleAttnHead(in_c, in_c//4, downsample_fac)
        self.final_conv = nn.Conv2d(2*in_c, in_c, 1)


    def forward(self, app_enc, app_pose_enc, pose_enc):
        attn_out_1 = self.attn_head_1(app_enc, app_pose_enc, pose_enc)
        attn_out_2 = self.attn_head_2(app_enc, app_pose_enc, pose_enc)
        attn_out_3 = self.attn_head_3(app_enc, app_pose_enc, pose_enc)
        attn_out_4 = self.attn_head_4(app_enc, app_pose_enc, pose_enc)
        out = torch.cat([pose_enc, attn_out_1, attn_out_2,
                         attn_out_3, attn_out_4], dim=1)
        out = self.final_conv(out)
        return out


class SingleAttnHead(nn.Module):
    def __init__(self, in_c, attn_size, downsample_fac=1):
        super().__init__()
        self.in_c = in_c
        self.attn_size = attn_size
        self.upsample = nn.Upsample(scale_factor=downsample_fac)

        self.q_conv = nn.Conv2d(in_c, attn_size, downsample_fac,
                                downsample_fac)
        self.k_conv = nn.Conv2d(in_c, attn_size, downsample_fac,
                                downsample_fac)
        self.v_conv = nn.Conv2d(in_c, attn_size, downsample_fac,
                                downsample_fac)

        self.gamma = nn.Parameter(torch.Tensor([0.]))

    def forward(self, app_enc, app_pose_enc, pose_enc):
        query = self.q_conv(pose_enc)
        key = self.k_conv(app_pose_enc)
        value = self.v_conv(app_enc)

        _, _, w, h = query.shape

        query = query.view(-1, self.attn_size, w*h).transpose(1, 2)
        key = key.view(-1, self.attn_size, w*h)
        value = value.view(-1, self.attn_size, w*h).transpose(1, 2)

        attn = query@key
        attn = attn / (self.attn_size**0.5)
        attn_out = attn@value
        out = attn_out.view(-1, w, h, self.attn_size).permute(0, 3, 1, 2)
        out = self.upsample(out)
        return self.gamma * out
