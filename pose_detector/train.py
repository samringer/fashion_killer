import pickle
from os.path import join

from absl import flags, app
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from apex import amp

from pose_detector.model.model import PoseDetector
from pose_detector.data_modules.dataset import PoseDetectorDataset
from pose_drawer.pose_drawer import Pose_Drawer
from util import (save_checkpoint,
                  load_checkpoint,
                  prepare_experiment_dirs,
                  get_tb_logger)

POSE_DRAWER = Pose_Drawer()

FLAGS = flags.FLAGS
flags.DEFINE_boolean('over_train', False, "Overtrain on one datapoint")
flags.DEFINE_float('learning_rate', 1e-4, "Starting learning rate")
flags.DEFINE_integer('batch_size', 4, "Batch size to use when training")
flags.DEFINE_integer('num_epochs', 10, "Number of training epochs")
flags.DEFINE_integer('min_joints_to_train_on', 10,
                     "The minimum num of joints an image should \
                      contain to be used as training data")


def train(unused_argv):
    # TODO: Assert that the flags are set correctly before doing this
    models_path = prepare_experiment_dirs()
    logger = get_tb_logger()

    model = PoseDetector()
    if FLAGS.use_cuda:
        model = model.cuda()

    dataset = PoseDetectorDataset(overtrain=FLAGS.over_train,
                                  min_joints_to_train_on=FLAGS.min_joints_to_train_on)
    dataloader = DataLoader(dataset, batch_size=FLAGS.batch_size,
                            shuffle=True, num_workers=4, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

    if FLAGS.use_fp16:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level='O1')

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    if FLAGS.load_checkpoint:
        model, optimizer = load_checkpoint(model, optimizer)
        print('Loaded from checkpoint {}'.format(FLAGS.load_checkpoint))

    for epoch in range(FLAGS.num_epochs):
        lr_scheduler.step()

        # Save a model at the start of each epoch
        save_path = join(models_path, '{}.pt'.format(epoch))
        torch.save(model.state_dict(), save_path)

        for i, batch in enumerate(dataloader):
            step_num = epoch*len(dataloader) + i
            _, pred_heat_maps, loss = _train_step(batch, model, optimizer)

            if step_num % FLAGS.tb_log_interval == 0:
                log_results(epoch, step_num, logger, pred_heat_maps, loss)
            if step_num % FLAGS.checkpoint_interval == 0:
                save_path = join(models_path, '{}.chk'.format(step_num))
                save_checkpoint(model, optimizer, save_path)
                print('Model & optimizer checkpointed at {}'.format(save_path))
            print(step_num, f"{loss.item():.3f}")


def _train_step(batch, model, optimizer):
    img = batch['img']
    true_heat_maps = batch['keypoint_heat_maps']
    true_pafs = batch['part_affinity_fields']
    kp_loss_mask = batch['kp_loss_mask']
    paf_loss_mask = batch['p_a_f_loss_mask']

    if FLAGS.use_cuda:
        img = img.cuda()
        true_heat_maps = true_heat_maps.cuda()
        true_pafs = true_pafs.cuda()
        kp_loss_mask = kp_loss_mask.cuda()
        paf_loss_mask = paf_loss_mask.cuda()

    pred_pafs, pred_heat_maps = model(img)

    # Need to scale down ground truth by factor of 4 to make dims match.
    true_heat_maps = nn.MaxPool2d(4)(true_heat_maps)
    true_pafs = nn.MaxPool2d(4)(true_pafs)

    hm_loss = get_heatmap_loss(pred_heat_maps, true_heat_maps,
                               kp_loss_mask)
    paf_loss = get_part_affinity_field_loss(pred_pafs, true_pafs,
                                            paf_loss_mask)
    loss = hm_loss + paf_loss

    optimizer.zero_grad()
    if FLAGS.use_fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    optimizer.step()

    return pred_pafs, pred_heat_maps, loss


def get_heatmap_loss(pred_heat_maps, true_heat_maps, kp_loss_mask):
    """
    Get MSE loss between true heatmap and pred heatmap.
    If joint is not present in the data then the corresponding
    loss using ignored using the keypoint loss mask.
    """
    heatmap_loss = 0
    kp_loss_mask = kp_loss_mask.unsqueeze(2).unsqueeze(3)
    true_heat_maps_masked = true_heat_maps * kp_loss_mask
    for pred_heat_map in pred_heat_maps:
        pred_heat_map_masked = pred_heat_map * kp_loss_mask
        heatmap_loss += nn.MSELoss()(pred_heat_map_masked, true_heat_maps_masked)
    return heatmap_loss


def get_part_affinity_field_loss(pred_pafs, true_pafs, paf_loss_mask):
    """
    Get MSE loss between true part affinity fields and
    predicted part affinity fields.
    If limb is not present in the data then the corresponding
    loss using ignored using the part affinity field loss mask.
    """
    paf_loss = 0
    paf_loss_mask = paf_loss_mask.unsqueeze(2).unsqueeze(3)
    true_pafs_masked = true_pafs * paf_loss_mask
    for pred_paf in pred_pafs:
        pred_paf_masked = pred_paf * paf_loss_mask
        paf_loss += nn.MSELoss()(pred_paf_masked, true_pafs_masked)
    return paf_loss


def log_results(epoch, step_num, writer, pred_heat_maps, loss):
    """
    Log the results using tensorboardx so they can be
    viewed using a tensorboard server.
    """
    # Interpolate up to make img a decent size.
    heat_maps = nn.functional.interpolate(pred_heat_maps[-1], scale_factor=4)
    heat_maps = heat_maps[0].cpu().detach().numpy()
    pose_img = POSE_DRAWER.draw_pose_from_heatmaps(heat_maps)
    pose_img = torch.Tensor(pose_img).permute(2, 0, 1)
    img_file_name = 'generated_pose/{}'.format(epoch)
    writer.add_image(img_file_name, pose_img, step_num)
    writer.add_scalar('Train/loss', loss, step_num)
    if step_num % 1000 == 0:
        with open('/tmp/{}.heatmap'.format(step_num), 'wb') as out_f:
            pickle.dump(pred_heat_maps, out_f)


if __name__ == "__main__":
    app.run(train)
