import argparse
from os import mkdir
from os.path import join, exists

import torch
from torch import nn, optim
from tensorboardX import SummaryWriter
from apex import amp
amp_handle = amp.init()

from Fashion_Killer.Model.Model import Model
import Fashion_Killer.hyperparams as hp
#from Model.Custom_VGG19 import get_custom_VGG19
from Fashion_Killer.DataLoader import get_Fashion_Killer_DataLoader


def train(exp_path):
    logger, models_path = _prepare_experiment_dirs(exp_path)
    data_loader, model, optimizer = _get_training_objects()

    for epoch in range(hp.num_epochs):
        torch.save(model.state_dict(), join(models_path, '{}.pt'.format(epoch)))
        for i, batch in enumerate(data_loader):
            step_num = (epoch*len(data_loader))+i
            gen_img, loss, l1_loss, kl_divergence = _training_step(batch, model, optimizer)
            print(loss)

            if step_num % hp.ts_log_interval == 0:
                log_results(epoch, step_num, logger, gen_img, loss, l1_loss, kl_divergence)


def _prepare_experiment_dirs(exp_path):
    if not exists(exp_path):
        mkdir(exp_path)

    models_path = join(exp_path, 'models')
    if not exists(models_path):
        mkdir(models_path)

    logs_path = join(exp_path, 'logs')
    if not exists(logs_path):
        mkdir(logs_path)

    logger = SummaryWriter(logs_path)

    return logger, models_path


def _get_training_objects():
    model = Model()
    #VGG = get_custom_VGG19().eval()
    if hp.use_cuda:
        #VGG = VGG.cuda()
        model = model.cuda()

    data_loader = get_Fashion_Killer_DataLoader(hp.bs)
    optimizer = optim.Adam(model.parameters(), lr=hp.learning_rate, betas=(hp.beta_1, hp.beta_2))
    return data_loader, model, optimizer


def _training_step(batch, model, optimizer):
    orig_img = batch['orig_img']
    pose_img = batch['pose_img']
    localised_joints = batch['localised_joints']

    if hp.use_cuda:
        orig_img = orig_img.cuda()
        pose_img = pose_img.cuda()
        localised_joints = localised_joints.cuda()

    # Original img pose and target pose are same for training
    gen_img, app_mu_1x1, app_mu_2x2, pose_mu_1x1, pose_mu_2x2 = model(orig_img, pose_img, pose_img, localised_joints)

    #VGG_loss = _get_VGG_loss(VGG, orig_img, gen_img)
    l1_loss = nn.L1Loss()(orig_img, gen_img)
    kl_divergence = _get_KL_Divergence(app_mu_1x1, app_mu_2x2, pose_mu_1x1, pose_mu_2x2)
    loss = l1_loss + kl_divergence
    #loss = VGG_loss + kl_divergence

    optimizer.zero_grad()
    if hp.use_fp16:
        with amp_handle.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    optimizer.step()

    return gen_img[0], loss, l1_loss, kl_divergence


def _get_KL_Divergence(app_mu_1x1, app_mu_2x2, pose_mu_1x1, pose_mu_2x2):
    term_1 = torch.sum(0.5 * ((app_mu_1x1 - pose_mu_1x1).pow(2)))
    term_2 = torch.sum(0.5 * ((app_mu_2x2 - pose_mu_2x2).pow(2)))
    return (term_1 + term_2) * hp.KL_weight


def _get_VGG_loss(VGG, orig_img, gen_img):
    orig_img_VGG_values_list = VGG(orig_img)
    gen_img_VGG_values_list = VGG(gen_img)

    total_l1_loss = 0
    for orig_VGG_value, gen_VGG_value, weight in zip(orig_img_VGG_values_list, gen_img_VGG_values_list, hp.feature_weights):
        l1_loss = nn.L1Loss()(orig_VGG_value, gen_VGG_value) * weight
        total_l1_loss += l1_loss

    return total_l1_loss


def log_results(epoch, step_num, writer, gen_img, loss, l1_loss, KL_Divergence):
    im_to_log = (gen_img * 0.5) + 0.5  # Undo effects of normalisation
    im_file_name = 'generated_images/{}'.format(epoch)
    writer.add_image(im_file_name, im_to_log, step_num)

    writer.add_scalar('Train/loss', loss, step_num)
    writer.add_scalar('Train/l1_loss', l1_loss, step_num)
    writer.add_scalar('Train/KL_Divergence', KL_Divergence, step_num)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', required=True)
    parser.add_argument('--root_log_dir', default='/home/sam/experiments/Fashion_Killer')
    args = parser.parse_args()

    exp_path = join(args.root_log_dir, args.exp_name)

    train(exp_path)
