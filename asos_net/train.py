from os.path import join

from absl import flags, app, logging
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from apex import amp

from asos_net.model.u_net import UNet
from asos_net.dataset import AsosDataset
from utils import (save_checkpoint,
                   load_checkpoint,
                   prepare_experiment_dirs,
                   get_tb_logger,
                   set_seeds)


FLAGS = flags.FLAGS

FLAGS.task_path = '/home/sam/experiments/AsosNet'
FLAGS.data_dir = '/home/sam/data/asos/train_clean'


def train(unused_argv):
    """
    Trains a network to be able to generate images from img/pose
    pairs of asos data.
    """
    models_path = prepare_experiment_dirs()
    logger = get_tb_logger()
    set_seeds()

    model = UNet()
    if FLAGS.use_cuda:
        model = model.cuda()

    dataset = AsosDataset(root_data_dir=FLAGS.data_dir,
                          overtrain=FLAGS.over_train)
    dataloader = DataLoader(dataset, batch_size=FLAGS.batch_size,
                            shuffle=True, num_workers=6, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

    if FLAGS.use_fp16:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level='O1')

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
            gen_imgs, loss = _train_step(batch, model, optimizer)

            if step_num % FLAGS.tb_log_interval == 0:
                log_results(epoch, step_num, logger, gen_imgs, loss)
            if step_num % FLAGS.checkpoint_interval == 0:
                save_checkpoint(model, optimizer, lr_scheduler, step_num)
            logging.info(f"{step_num} {loss.item():.4f}")
            lr_scheduler.step(loss.item())
            step_num += 1


def _train_step(batch, model, optimizer):
    app_img = batch['app_img']
    pose_img = batch['pose_img']
    target_img = batch['target_img']

    if FLAGS.use_cuda:
        app_img = app_img.cuda()
        pose_img = pose_img.cuda()
        target_img = target_img.cuda()

    gen_img = model(app_img, pose_img)
    loss = nn.L1Loss()(gen_img, target_img)

    optimizer.zero_grad()
    if FLAGS.use_fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    optimizer.step()

    return gen_img, loss


def log_results(epoch, step_num, writer, gen_img, loss):
    """
    Log the results using tensorboardx so they can be
    viewed using a tensorboard server.
    """
    gen_img = gen_img[0].detach().cpu()
    img_file_name = 'generated_img/{}'.format(epoch)
    writer.add_image(img_file_name, gen_img, step_num)
    writer.add_scalar('Train/loss', loss, step_num)


if __name__ == "__main__":
    app.run(train)
