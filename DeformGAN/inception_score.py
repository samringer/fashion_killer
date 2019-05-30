# Based on https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader

from torchvision.models.inception import inception_v3
from torchvision import transforms

import numpy as np
from scipy.stats import entropy

from DeformGAN.dataset import AsosDataset
from utils import set_seeds


def normalize(batch):
    batch = batch-0.5
    batch = batch*0.5
    return batch


def score_validation():
    set_seeds()
    val_dataset = AsosDataset(root_data_dir='/home/sam/data/asos/2604_clean/val')
    val_dataloader = DataLoader(val_dataset, batch_size=64)

    model = inception_v3(pretrained=True, transform_input=False)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    inception_scores = []
    for batch in val_dataloader:
        imgs = normalize(batch['app_img'])
        if torch.cuda.is_available():
            imgs = imgs.cuda()
        inception_scores.append(inception_score_on_batch(imgs, model))

    return sum(inception_scores)/len(inception_scores)


def inception_score_on_batch(x, model):
    x = F.interpolate(x, size=(299, 299))
    x = model(x)
    model_out = F.softmax(x, dim=1).data.cpu().numpy()
    p_y = np.mean(model_out, axis=0)
    scores = [entropy(p_yx, p_y) for p_yx in model_out]
    return np.exp(np.mean(scores))


if __name__ == '__main__':
    print(score_validation())
