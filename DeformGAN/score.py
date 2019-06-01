# Based on https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py

import argparse

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

from DeformGAN.dataset import AsosDataset
from DeformGAN.model.generator import Generator
from DeformGAN.perceptual_loss_vgg import PerceptualLossVGG
from utils import set_seeds


def score_generator_on_validation(generator):
    set_seeds()
    val_dataset = AsosDataset(root_data_dir='/home/sam/data/asos/2604_clean/val')
    val_dataloader = DataLoader(val_dataset, batch_size=64)

    inception_model = inception_v3(pretrained=True,
                                   transform_input=False)
    perceptual_loss_vgg = PerceptualLossVGG()
    if torch.cuda.is_available():
        inception_model = inception_model.cuda()
        generator = generator.cuda()
        perceptual_loss_vgg = perceptual_loss_vgg.cuda()
    inception_model.eval()
    perceptual_loss_vgg.eval()
    generator.eval()

    total_l1_loss = 0
    total_perceptual_loss = 0
    inception_scores = []
    for batch in val_dataloader:
        app_img = batch['app_img']
        app_pose_img = batch['app_pose_img']
        target_img = batch['target_img']
        pose_img = batch['pose_img']

        # Downsample as in traintime
        app_img = nn.MaxPool2d(kernel_size=4)(app_img)
        app_pose_img = nn.MaxPool2d(kernel_size=4)(app_pose_img)
        target_img = nn.MaxPool2d(kernel_size=4)(target_img)
        pose_img = nn.MaxPool2d(kernel_size=4)(pose_img)

        if torch.cuda.is_available():
            app_img = app_img.cuda()
            app_pose_img = app_pose_img.cuda()
            target_img = target_img.cuda()
            pose_img = pose_img.cuda()

        with torch.no_grad():
            gen_img = generator(app_img, app_pose_img, pose_img)
            total_l1_loss += nn.L1Loss()(app_img, gen_img).item()
            total_perceptual_loss += perceptual_loss_vgg(gen_img,
                                                         target_img)
        batch_score = inception_score_on_batch(gen_img, inception_model)
        inception_scores.append(batch_score)

    inception_score = sum(inception_scores)/len(inception_scores)
    l1_loss = total_l1_loss/len(val_dataloader)
    perceptual_loss = total_perceptual_loss/len(val_dataloader)
    return inception_score, l1_loss, perceptual_loss


def inception_score_on_batch(x, inception_model):
    x = F.interpolate(x, size=(299, 299))
    with torch.no_grad():
        x = inception_model(x)
    model_out = F.softmax(x, dim=1).data.cpu().numpy()
    p_y = np.mean(model_out, axis=0)
    scores = [entropy(p_yx, p_y) for p_yx in model_out]
    return np.exp(np.mean(scores))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("generator_path")
    args = parser.parse_args()
    generator = Generator()
    generator.load_state_dict(torch.load(args.generator_path))
    inception_score, l1_loss, perceptual_loss = score_generator_on_validation(generator)
    print(f"Inception score: {inception_score:.3f}")
    print(f"L1 loss: {l1_loss:.3f}")
    print(f"Perceptual loss: {perceptual_loss:.3f}")
