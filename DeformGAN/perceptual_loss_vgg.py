# Based on https://github.com/jantic/DeOldify/blob/master/fasterai/loss.py

import torch
from torch import nn
from torchvision.models.vgg import vgg16_bn


class PerceptualLossVGG(nn.Module):
    def __init__(self):
        super().__init__()

        self.vgg = vgg16_bn(True).features
        blocks = [i-1 for i, o in enumerate(self.vgg.children())
                  if isinstance(o, nn.MaxPool2d)]
        self.layer_ids = blocks[2:5]
        self.wgts = [20, 70, 10]

    def get_features(self, x):
        features = []
        for i, block in enumerate(self.vgg):
            x = block(x)
            if i in self.layer_ids:
                features.append(x)
            if i >= self.layer_ids[-1]:
                break
        return features

    def forward(self, x, target):
        x_feats = self.get_features(x)
        with torch.no_grad():
            target_feats = self.get_features(target)
        feat_losses = []
        for x_feat, t_feat, wgt in zip(x_feats, target_feats, self.wgts):
            feat_losses += [nn.L1Loss()(x_feat, t_feat)*wgt]
        return sum(feat_losses)
