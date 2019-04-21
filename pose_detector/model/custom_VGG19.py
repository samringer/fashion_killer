import os

import torch
from torch import nn
from torch.utils import model_zoo
from torchvision.models.vgg import VGG, model_urls

full_layer_spec = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
                   512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']


class Custom_VGG19(VGG):
    """
    A VGG19 model that only uses the first 10 layers.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i >= 10:
                break
        return x


def get_custom_VGG19(pretrained=True, **kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = Custom_VGG19(make_layers(full_layer_spec), **kwargs)
    if pretrained:
        if os.path.isfile('pretrained_models/vgg19.pt'):
            model.load_state_dict(torch.load('pretrained_models/vgg19.pt'))
        else:
            # Model weights not found so download from model zoo.
            model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
