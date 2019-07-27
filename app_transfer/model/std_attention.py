import torch
from torch import nn

from app_transfer.model.spectral_norm import SpecNorm
# Based on: https://github.com/christiancosgrove/pytorch-sagan/


class AttentionMech(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.in_c = in_c
        self.attn_c = in_c//4
        self.q_conv = nn.Conv2d(in_c, self.attn_c, 1)
        self.k_conv = nn.Conv2d(in_c, self.attn_c, 1)
        self.v_conv = nn.Conv2d(in_c, in_c, 1)
        self.gamma = nn.Parameter(torch.Tensor([0.]))  # init as 0

    def forward(self, x):
        query = self.q_conv(x)
        key = self.k_conv(x)
        value = self.v_conv(x)

        _, _, w, h = query.shape

        query = query.view(-1, self.attn_c, w*h).transpose(1, 2)
        key = key.view(-1, self.attn_c, w*h)
        value = value.view(-1, self.in_c, w*h).transpose(1, 2)

        attn = query@key
        attn = nn.Softmax(dim=1)(attn)
        attn_out = attn@value
        attn_out = attn_out.view(-1, w, h, self.in_c).permute(0, 3, 1, 2)
        out = self.gamma*attn_out + x
        return out
