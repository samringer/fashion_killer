import argparse
import pickle
from os import mkdir
from os.path import join, exists

import torch
from torch import nn, optim
from tensorboardX import SummaryWriter
from apex import amp
amp_handle = amp.init()

from pose_detector.model.model import Model
import pose_detector.hyperparams as hp
from pose_detector.data_modules.dataloader import Pose_Detector_DataLoader
from pose_drawer.pose_drawer import Pose_Drawer

POSE_DRAWER = Pose_Drawer()


def train(exp_path):
    logger, models_path = _prepare_experiment_dirs(exp_path)
    data_loader, model, optimizer = _get_training_objects()

    for epoch in range(hp.num_epochs):
        torch.save(model.state_dict(), join(models_path, '{}.pt'.format(epoch)))
        for i, batch in enumerate(data_loader):
            step_num = (epoch*len(data_loader))+i
            pred_heat_maps, loss = _training_step(batch, model, optimizer)
            print(loss)

            if step_num % hp.ts_log_interval == 0:
                log_results(epoch, step_num, logger, pred_heat_maps, loss)


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
    if hp.use_cuda:
        model = model.cuda()

    data_loader = Pose_Detector_DataLoader(hp.batch_size, overtrain=hp.overtrain, min_joints_to_train_on=hp.min_joints_to_train_on)

    optimizer = optim.Adam(model.parameters(), lr=hp.learning_rate, betas=(hp.beta_1, hp.beta_2))

    return data_loader, model, optimizer


def _training_step(batch, model, optimizer):
    img = batch['img']
    true_heat_maps = batch['keypoint_heat_maps']
    loss_mask = batch['loss_mask']

    if hp.use_cuda:
        img = img.cuda()
        true_heat_maps = true_heat_maps.cuda()
        loss_mask = loss_mask.cuda()

    pred_heat_maps = model(img)

    # Need to scale down ground truth
    # by a factor of 4 to make dims match.
    true_heat_maps = nn.MaxPool2d(4)(true_heat_maps)

    loss = get_heatmap_loss(pred_heat_maps, true_heat_maps, loss_mask)

    optimizer.zero_grad()
    if hp.use_fp16:
        with amp_handle.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    optimizer.step()

    return pred_heat_maps, loss


def get_heatmap_loss(pred_heat_maps, true_heat_maps, loss_mask):
    """
    Get MSE loss between true heatmap and pred heatmap.
    If joint is not present in the data then the corresponding
    loss using ignored using the loss mask.
    """
    heatmap_loss = 0
    loss_mask = loss_mask.unsqueeze(2).unsqueeze(3)
    true_heat_maps_masked = true_heat_maps * loss_mask
    for pred_heat_map in pred_heat_maps:
        pred_heat_map_masked = pred_heat_map * loss_mask
        heatmap_loss += nn.MSELoss()(pred_heat_map_masked, true_heat_maps_masked)
    return heatmap_loss


def log_results(epoch, step_num, writer, pred_heat_maps, loss):
    # Interpolate up to make img a decent size.
    heat_maps = nn.functional.interpolate(pred_heat_maps[2], scale_factor=4)
    heat_maps = heat_maps[0].cpu().detach().numpy()
    heat_map_list = [heat_maps[i] for i in range(heat_maps.shape[0])]
    pose_img = POSE_DRAWER.draw_pose_from_heatmaps(heat_map_list)
    pose_img = torch.Tensor(pose_img).permute(2, 0, 1)
    img_file_name = 'generated_pose/{}'.format(epoch)
    writer.add_image(img_file_name, pose_img, step_num)
    writer.add_scalar('Train/loss', loss, step_num)
    if step_num % 1000 == 0:
        with open('/tmp/{}.heatmap'.format(step_num), 'wb') as out_f:
            pickle.dump(pred_heat_maps, out_f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', required=True)
    parser.add_argument('--root_log_dir', default='/home/sam/experiments/Pose_Detector')
    args = parser.parse_args()

    exp_path = join(args.root_log_dir, args.exp_name)

    train(exp_path)
