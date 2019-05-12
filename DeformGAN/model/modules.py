import torch
from torch import nn


class GenEncConvLayer(nn.Module):
    def __init__(self, in_c, out_c, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, stride=stride)
        self.instance_norm = nn.InstanceNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.instance_norm(x)
        x = nn.ReLU()(x)
        return x


class GenDecConvLayer(nn.Module):
    def __init__(self, in_c, out_c, stride=2, dropout=False):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, stride=stride)
        self.instance_norm = nn.InstanceNorm2d(out_c)
        self.dropout = dropout

    def forward(self, source_enc_f, target_enc_f, prev_inp=None):
        if prev_inp is not None:
            x = torch.cat([source_enc_f, target_enc_f, prev_inp], dim=1)
        else:
            x = torch.cat([source_enc_f, target_enc_f], dim=1)
        if self.dropout:
            x = nn.Dropout()(x)
        x = self.conv(x)
        x = self.instance_norm(x)
        x = nn.ReLU()(x)
        return x


class DisConvLayer(nn.Module):
    def __init__(self, in_c, out_c, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, stride=stride)

    def forward(self, x):
        x = self.conv(x)
        x = nn.ReLU()(x)
        return x
