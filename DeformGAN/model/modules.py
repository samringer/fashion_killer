import torch
import torch.utils.checkpoint as chk
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


class GenDecAttnBlock(nn.Module):
    def __init__(self, in_c, prev_in_c, out_c, dropout=False):
        super().__init__()
        self.attn_mech = AttnMech(in_c)
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
        #x = self.attn_mech(source_enc_f, target_enc_f)
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
    def __init__(self, in_c):
        super().__init__()
        self.attn_size = in_c//8
        self.attn_out = self.attn_size
        self.key_conv = nn.Conv2d(in_c, self.attn_size, 1)
        self.query_conv = nn.Conv2d(in_c, self.attn_size, 1)
        self.k_q_linear = nn.Linear(self.attn_size*2, self.attn_out)
        self.attn_conv = nn.Conv2d(self.attn_out+in_c, in_c, 1)
        self.gamma = nn.Parameter(torch.Tensor([0.]))

    def forward(self, source_enc_f, target_enc_f):
        _, _, width, height = source_enc_f.shape

        query = self.query_conv(source_enc_f)
        query = query.view(-1, self.attn_size, width*height).transpose(1, 2)

        key = self.key_conv(target_enc_f)
        key = key.view(-1, self.attn_size, width*height).transpose(1, 2)

        query = query.unsqueeze(1).expand(-1, width*height, -1, -1)
        query = query.contiguous().view(-1, (width*height)**2, self.attn_size)
        key = key.unsqueeze(2).expand(-1, -1, width*height, -1)
        key = key.contiguous().view(-1, (width*height)**2, self.attn_size)

        x = torch.cat([key, query], dim=2)
        x = self.k_q_linear(x)
        x = x.view(-1, width*height, width*height, self.attn_out)
        x = x.sum(dim=2)
        x = x.transpose(1, 2).view(-1, self.attn_out, width, height)
        x = torch.cat([target_enc_f, x], dim=1)
        out = self.attn_conv(x)
        return target_enc_f + (self.gamma*out)


class DisConvLayer(nn.Module):
    def __init__(self, in_c, out_c, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, stride=stride)

    def forward(self, x):
        x = self.conv(x)
        x = nn.ReLU()(x)
        return x
