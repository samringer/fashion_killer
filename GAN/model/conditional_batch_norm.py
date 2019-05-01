#Reference: https://github.com/pytorch/pytorch/issues/8985

import torch
from torch import nn


class ConditionalBatchNorm2d(nn.Module):
    """
    Normal batch norm: gamma and beta are std learnable params
    Cond BN: gamma and beta are learnable params that depend
             on the class of input
    """
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)
        # Initialise bias at 0
        self.embed.weight.data[:, num_features:].zero_()

    def forward(self, x, y):
        x = self.bn(x)
        # Chunk cuts tensor down middle
        gamma, beta = self.embed(y).chunk(2, 1)
        x = gamma.view(-1, self.num_features, 1, 1) * x + beta.view(-1, self.num_features, 1, 1)
        return x
