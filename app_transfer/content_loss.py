import torch
from torch import nn
from torchvision.models.vgg import vgg16


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()

    def forward(self, gen_images, target_images):
        # Perception Loss
        return self.mse_loss(self.loss_network(gen_images), self.loss_network(target_images))
