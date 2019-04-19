from torch import nn

num_joints = 18
num_limbs = 17

class LimbBlock(nn.Module):

    def __init__(self, in_c):
        super().__init__()
        self.conv_block_1 = ConvBlock(in_c)
        self.conv_block_2 = ConvBlock(in_c)
        self.conv_block_3 = ConvBlock(in_c)
        self.conv_block_4 = ConvBlock(in_c)
        self.conv_block_5 = ConvBlock(in_c)
        self.conv_1x1a = ConvLayer(in_c, in_c, kernel_size=1, padding=0)
        self.conv_1x1b = ConvLayer(in_c, num_limbs*2,
                                    kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.conv_block_5(x)
        x = self.conv_1x1a(x)
        paf = self.conv_1x1b(x)
        return paf


class JointBlock(nn.Module):

    def __init__(self, in_c):
        super().__init__()
        self.conv_block_1 = ConvBlock(in_c)
        self.conv_block_2 = ConvBlock(in_c)
        self.conv_block_3 = ConvBlock(in_c)
        self.conv_block_4 = ConvBlock(in_c)
        self.conv_block_5 = ConvBlock(in_c)
        self.conv_1x1a = ConvLayer(in_c, in_c, kernel_size=1, padding=0)
        self.conv_1x1b = ConvLayer(in_c, num_joints, kernel_size=1, padding=0)

    def forward(self, x_0):
        x = self.conv_block_1(x_0)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.conv_block_5(x)
        x = self.conv_1x1a(x)
        x = self.conv_1x1b(x)
        # TODO: Add back in
        # heat_maps = nn.Sigmoid()(x)
        heat_maps = x
        return heat_maps


class ConvBlock(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.conv_1 = ConvLayer(in_c, in_c)
        self.conv_2 = ConvLayer(in_c, in_c)
        self.conv_3 = ConvLayer(in_c, in_c)

    def forward(self, x_0):
        x_1 = self.conv_1(x_0)
        x_2 = self.conv_2(x_1)
        x_3 = self.conv_3(x_2)
        out = x_0 + x_1 + x_2 + x_3
        return out


class ConvLayer(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, padding=padding,
                              bias=False)
        self.l_relu = nn.LeakyReLU(negative_slope=0.2)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.l_relu(x)
        x = self.bn(x)
        return x
