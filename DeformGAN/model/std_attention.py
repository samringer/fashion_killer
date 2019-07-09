import torch
from torch import nn

# Based on: https://github.com/christiancosgrove/pytorch-sagan/


class AttentionMech(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.value = nn.Conv2d(in_c, in_c, 1)
        self.self_attn = SelfAttention(in_c)
        self.gamma = nn.Parameter(torch.zeros(1)) #init as 0

    def forward(self, x):
        batch_size, in_c, width, height = x.size()
        value = self.value(x).view(batch_size, in_c, width*height)
        self_attn = self.self_attn(x)
        out = value @ self_attn.transpose(1, 2)
        out = out.view(batch_size, in_c, width, height)
        out = self.gamma*out + x
        return out


class SelfAttention(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        attn_c = in_c//8
        self.query = nn.Conv2d(in_c, attn_c, 1)
        self.key = nn.Conv2d(in_c, attn_c, 1)

    def forward(self, x):
        batch_size, _, width, height = x.size()
        query = self.query(x).view(batch_size, -1, width*height)
        key = self.key(x).view(batch_size, -1, width*height)
        energy = query.transpose(1, 2) @ key
        attn = nn.Softmax(dim=1)(energy)
        return attn
