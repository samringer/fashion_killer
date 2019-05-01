import torch
from torch import nn
from GAN.model.spectral_norm import SpecNorm

# Based on: https://github.com/christiancosgrove/pytorch-sagan/


class AttentionMech(nn.Module):

    def __init__(self, in_c):
        super().__init__()
        self.value = SpecNorm(nn.Conv2d(in_c, in_c, 1))
        self.self_attn = SelfAttention(in_c)
        self.gamma = nn.Parameter(torch.zeros(1)) #init as 0

    def forward(self, x):
        bs, in_c, w, h = x.size()
        value = self.value(x).view(bs, in_c, w*h)
        self_attn = self.self_attn(x)
        out = value @ self_attn.transpose(1, 2)
        out = out.view(bs, in_c, w, h)
        out = self.gamma*out + x
        return out


class SelfAttention(nn.Module):

    def __init__(self, in_c):
        super().__init__()
        attn_c = in_c//8
        self.query = SpecNorm(nn.Conv2d(in_c, attn_c, 1))
        self.key = SpecNorm(nn.Conv2d(in_c, attn_c, 1))

    def forward(self, x):
        bs, in_c, w, h = x.size()
        query = self.query(x).view(bs, -1, w*h)
        key = self.key(x).view(bs, -1, w*h)
        energy = query.transpose(1, 2) @ key
        attn = nn.Softmax(dim=1)(energy)
        return attn
