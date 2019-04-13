import torch
from torch import nn
import v_u_net.hyperparams as hp


class EncoderBlock(nn.Module):
    """
    We need to return both x_1 and x_2 as the Unet uses
    both for it's long skip connections
    (The appearance encoder does not need x_1)
    """

    def __init__(self, in_c, out_c):
        super().__init__()
        self.res_block = EncResLayer(in_c)
        self.conv = nn.Conv2d(in_c, out_c, 3, stride=2, padding=1) # Performs downsampling

    def forward(self, x):
        x_1 = self.res_block(x)
        x_2 = self.conv(x_1)
        return x_1, x_2


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.res_layer = DecResLayer(in_c*2)
        self.conv = nn.Conv2d(in_c*2, out_c, 3, padding=1)

    def forward(self, x, skip_con_1, skip_con_2):
        x = torch.cat((x, skip_con_1), dim=1)
        x = self.res_layer(x)
        x = torch.cat((x, skip_con_2), dim=1)
        x = nn.functional.interpolate(x, scale_factor=2) # Performs upsampling
        x = self.conv(x)
        return x


class EncResLayer(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.l_relu = nn.LeakyReLU(negative_slope=hp.leakiness)
        self.conv_1 = nn.Conv2d(in_c, in_c, 3, padding=1)
        self.conv_2 = nn.Conv2d(in_c, in_c, 3, padding=1)

    def forward(self, x):
        res = x
        x = self.l_relu(x)
        x = self.conv_1(x)
        x = self.l_relu(x)
        x = self.conv_2(x)
        return res + x


class DecResLayer(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.proj = nn.Conv2d(in_c, in_c//2, 1)
        self.conv_1 = nn.Conv2d(in_c, in_c, 3, padding=1)
        self.conv_2 = nn.Conv2d(in_c, in_c//2, 3, padding=1)
        self.l_relu = nn.LeakyReLU(negative_slope=hp.leakiness)

    def forward(self, x):
        res = x
        x = self.l_relu(x)
        x = self.conv_1(x)
        x = self.l_relu(x)
        x = self.conv_2(x)
        out = x + self.proj(res)
        return out
