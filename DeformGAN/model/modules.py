import torch
from torch import nn


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


class GenDecConvLayer(nn.Module):
    def __init__(self, in_c, out_c, stride=1, dropout=False):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1)
        self.instance_norm = nn.InstanceNorm2d(out_c)
        self.dropout = dropout

    def forward(self, source_enc_f, target_enc_f, prev_inp):
        x = nn.functional.interpolate(prev_inp, scale_factor=2)
        x = torch.cat([source_enc_f, target_enc_f, x], dim=1)
        if self.dropout:
            x = nn.Dropout()(x)
        x = self.conv(x)
        x = self.instance_norm(x)
        x = nn.ReLU()(x)
        return x


class GenDecAttnBlock(nn.Module):
    def __init__(self, in_c, prev_in_c, out_c, stride=1, dropout=False):
        super().__init__()
        self.attn_mech = AttnMech(in_c)
        conv_in = in_c + prev_in_c
        self.conv = nn.Conv2d(conv_in, out_c, 3, stride=stride,
                              padding=1)
        self.instance_norm = nn.InstanceNorm2d(out_c)
        self.dropout = dropout

    def forward(self, source_enc_f, target_enc_f, prev_inp):
        prev_inp = nn.functional.interpolate(prev_inp, scale_factor=2)
        x = self.attn_mech(source_enc_f, target_enc_f)
        x = torch.cat([x, prev_inp], dim=1)
        if self.dropout:
            x = nn.Dropout()(x)
        x = self.conv(x)
        x = self.instance_norm(x)
        x = nn.ReLU()(x)
        return x


class AttnMech(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.attn_size = in_c//2
        self.key_conv = nn.Conv2d(in_c, self.attn_size, 1)
        self.query_conv = nn.Conv2d(in_c, self.attn_size, 1)
        self.value_conv = nn.Conv2d(in_c, self.attn_size, 1)
        self.proj_up_conv = nn.Conv2d(self.attn_size, in_c, 1)
        self.gamma = nn.Parameter(torch.Tensor([0.]))

    def forward(self, source_enc_f, target_enc_f):
        _, _, width, height = source_enc_f.shape

        query = self.query_conv(source_enc_f)
        key = self.key_conv(target_enc_f)
        value = self.value_conv(target_enc_f)

        key = key.view(-1, self.attn_size, width*height).transpose(1, 2)
        query = query.view(-1, self.attn_size, width*height)
        attn = nn.Softmax(dim=1)(key@query)

        value = value.permute(0, 2, 3, 1).contiguous()
        value = value.view(-1, width*height, self.attn_size)
        out = attn@value
        out = out.view(-1, width, height, self.attn_size).permute(0, 3, 1, 2)
        out = self.proj_up_conv(out)
        return target_enc_f + (self.gamma*out)


class DisConvLayer(nn.Module):
    def __init__(self, in_c, out_c, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, stride=stride)

    def forward(self, x):
        x = self.conv(x)
        x = nn.ReLU()(x)
        return x
