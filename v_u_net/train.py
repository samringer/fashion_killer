from os.path import join

from absl import flags, app, logging
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from apex import amp

from v_u_net.model.v_u_net import VUNet
from v_u_net.dataset import VUNetDataset
from utils import (save_checkpoint,
                   load_checkpoint,
                   prepare_experiment_dirs,
                   get_tb_logger,
                   set_seeds)

FLAGS = flags.FLAGS


def train(unused_argv):
    """
    Trains a variational u-net for appearance transfer.
    """
    models_path = prepare_experiment_dirs()
    logger = get_tb_logger()
    set_seeds()

    model = VUNet()
    if FLAGS.use_cuda:
        model = model.cuda()

    dataset = VUNetDataset(root_data_dir=FLAGS.data_dir,
                           overtrain=FLAGS.over_train)
    dataloader = DataLoader(dataset, batch_size=FLAGS.batch_size,
                            shuffle=True, num_workers=6, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

    if FLAGS.use_fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    step_num = 0
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5,
                                     patience=5000, verbose=True,
                                     min_lr=FLAGS.learning_rate/50)

    if FLAGS.load_checkpoint:
        checkpoint_state = load_checkpoint(model, optimizer, lr_scheduler)
        model, optimizer, lr_scheduler, step_num = checkpoint_state

    for epoch in range(FLAGS.num_epochs):

        # Save a model at the start of each epoch
        save_path = join(models_path, '{}.pt'.format(epoch))
        torch.save(model.state_dict(), save_path)

        for batch in dataloader:
            gen_img, loss, l1_loss, kl_divergence = _train_step(batch, model, optimizer)

            if step_num % FLAGS.tb_log_interval == 0:
                log_results(epoch, step_num, logger, gen_img, loss, l1_loss, kl_divergence)

            if step_num % FLAGS.checkpoint_interval == 0:
                save_checkpoint(model, optimizer, lr_scheduler, step_num)

            logging.info(f"{step_num} {loss.item():.4f}")
            lr_scheduler.step(loss.item())
            step_num += 1


def _train_step(batch, model, optimizer):
    orig_img = batch['app_img']
    pose_img = batch['pose_img']
    localised_joints = batch['localised_joints']

    if FLAGS.use_cuda:
        orig_img = orig_img.cuda()
        pose_img = pose_img.cuda()
        localised_joints = localised_joints.cuda()

    # Original img pose and target pose are same for training
    model_out = model(orig_img, pose_img, pose_img, localised_joints)
    gen_img, app_mu_1x1, app_mu_2x2, pose_mu_1x1, pose_mu_2x2 = model_out

    l1_loss = nn.L1Loss()(orig_img, gen_img)
    kl_divergence = _get_KL_Divergence(app_mu_1x1, app_mu_2x2,
                                       pose_mu_1x1, pose_mu_2x2)

    loss = kl_divergence + l1_loss

    optimizer.zero_grad()
    if FLAGS.use_fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    optimizer.step()

    return gen_img[0], loss, l1_loss, kl_divergence


def _get_KL_Divergence(app_mu_1x1, app_mu_2x2, pose_mu_1x1, pose_mu_2x2):
    term_1 = torch.sum(0.5 * ((app_mu_1x1 - pose_mu_1x1).pow(2)))
    term_2 = torch.sum(0.5 * ((app_mu_2x2 - pose_mu_2x2).pow(2)))
    kl_divergence = term_1 + term_2
    # Scaling factor is empirical
    return 1e-5 * kl_divergence


def log_results(epoch, step_num, writer, gen_img, loss, l1_loss, KL_Divergence):

    img_file_name = 'generated_images/{}'.format(epoch)
    writer.add_image(img_file_name, gen_img, step_num)

    writer.add_scalar('Train/loss', loss, step_num)
    writer.add_scalar('Train/l1_loss', l1_loss, step_num)
    writer.add_scalar('Train/KL_Divergence', KL_Divergence, step_num)


if __name__ == "__main__":
    app.run(train)
