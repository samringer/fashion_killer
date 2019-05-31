from os.path import join
import pickle

from absl import flags, app, logging
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from apex import amp

from GAN.model.u_net import UNet
from GAN.model.generator import Generator
from GAN.model.discriminator import Discriminator
from GAN.dataset import AsosDataset
from utils import (prepare_experiment_dirs,
                   get_tb_logger,
                   set_seeds)


FLAGS = flags.FLAGS

flags.DEFINE_float('discriminator_lr', 1e-4, "Starting learning rate of the disciminator")


def train(unused_argv):
    """
    Trains a network to be able to generate images from img/pose
    pairs of asos data.
    """
    models_path = prepare_experiment_dirs()
    logger = get_tb_logger()
    set_seeds()

    generator = UNet()
    discriminator = Discriminator()
    if FLAGS.use_cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    dataset = AsosDataset(root_data_dir=FLAGS.data_dir,
                          overtrain=FLAGS.over_train)
    dataloader = DataLoader(dataset, batch_size=FLAGS.batch_size,
                            shuffle=True, num_workers=6, pin_memory=True)

    g_optimizer = optim.Adam(generator.parameters(), lr=FLAGS.learning_rate, betas=(0.0, 0.9))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=FLAGS.discriminator_lr, betas=(0.0, 0.9))

    if FLAGS.use_fp16:
        generator, g_optimizer = amp.initialize(generator, g_optimizer,
                                                opt_level='O1')
        discriminator, d_optimizer = amp.initialize(discriminator, d_optimizer,
                                                    opt_level='O1')

    step_num = 0
    # TODO: Add back in
    #lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5,
    #                                 patience=5000, verbose=True,
    #                                 min_lr=FLAGS.learning_rate/50)

    if FLAGS.load_checkpoint:
        checkpoint_state = load_checkpoint()
        generator.load_state_dict(checkpoint_state['generator'])
        discriminator.load_state_dict(checkpoint_state['discriminator'])
        g_optimizer.load_state_dict(checkpoint_state['g_optimizer'])
        d_optimizer.load_state_dict(checkpoint_state['d_optimizer'])
        step_num = checkpoint_state['step_num']

    for epoch in range(FLAGS.num_epochs):

        # Save a generator at the start of each epoch
        save_path = join(models_path, '{}.pt'.format(epoch))
        torch.save(generator.state_dict(), save_path)

        iter_dataloader = iter(dataloader)
        # Trick to use a larger effective batch size.
        for _ in range(len(dataloader)//2):
            batch_1 = next(iter_dataloader)
            batch_2 = next(iter_dataloader)
            batch_1_data = _prepare_batch_data(batch_1)
            batch_2_data = _prepare_batch_data(batch_2)

            d_optimizer.zero_grad()
            for batch_data in [batch_1_data, batch_1_data]:
                app_img, pose_img, target_img = batch_data

                with torch.no_grad():
                    gen_img = generator(app_img, pose_img)

                # Hinge loss
                d_fake_score = discriminator(app_img, pose_img, gen_img)
                d_real_score = discriminator(app_img, pose_img, target_img)

                d_fake_loss = nn.ReLU()(1 + d_fake_score)
                d_real_loss = nn.ReLU()(1 - d_real_score)
                d_loss = d_fake_loss.mean() + d_real_loss.mean()
                if FLAGS.use_fp16:
                    with amp.scale_loss(d_loss, d_optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    d_loss.backward()
            d_optimizer.step()

            g_optimizer.zero_grad()
            for batch_data in [batch_1_data, batch_2_data]:
                app_img, pose_img, target_img = batch_data
                gen_img = generator(app_img, pose_img)
                l1_loss = 5 * nn.L1Loss()(gen_img, target_img)
                # TODO: Why are we sending all three images through here?
                gan_loss = - discriminator(app_img, pose_img, gen_img).mean()
                g_loss = l1_loss + gan_loss
                if FLAGS.use_fp16:
                    with amp.scale_loss(g_loss, g_optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    g_loss.backward()
            g_optimizer.step()

            if step_num % FLAGS.tb_log_interval == 0:
                log_results(epoch, step_num, logger, gen_img,
                           d_loss, g_loss, l1_loss, gan_loss)

            if step_num % FLAGS.checkpoint_interval == 0:
                # TODO: Add in lr scheduler
                checkpoint_state = {
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    'step_num': step_num
                }
                save_checkpoint(checkpoint_state)
            logging.info(f"{step_num} {d_loss.item():.4f} {g_loss.item():.4f}")
            #logging.info(f"{step_num} {l1_loss.item():.4f}")
            #lr_scheduler.step(loss.item())
            step_num += 1


def _prepare_batch_data(batch):
    app_img = batch['app_img']
    pose_img = batch['pose_img']
    target_img = batch['target_img']

    if FLAGS.use_cuda:
        app_img = app_img.cuda()
        pose_img = pose_img.cuda()
        target_img = target_img.cuda()

    app_img = nn.MaxPool2d(kernel_size=2)(app_img)
    pose_img = nn.MaxPool2d(kernel_size=2)(pose_img)
    target_img = nn.MaxPool2d(kernel_size=2)(target_img)

    return app_img, pose_img, target_img


def log_results(epoch, step_num, writer, gen_img, d_loss, g_loss, l1_loss, gan_loss):
    """
    Log the results using tensorboardx so they can be
    viewed using a tensorboard server.
    """
    gen_img = gen_img[0].detach().cpu()
    img_file_name = 'generated_img/{}'.format(epoch)
    writer.add_image(img_file_name, gen_img, step_num)
    writer.add_scalar('Train/g_loss', g_loss, step_num)
    writer.add_scalar('Train/l1_loss', l1_loss, step_num)
    writer.add_scalar('Train/gan_loss', gan_loss, step_num)
    writer.add_scalar('Train/d_loss', d_loss, step_num)


def save_checkpoint(checkpoint_state):
    """
    Checkpoint the train_state.
    Saves them all together in a tuple representing the train state.
    """
    exp_dir = join(FLAGS.task_path, FLAGS.exp_name)
    step_num = checkpoint_state['step_num']
    save_path = join(exp_dir, 'models', '{}.chk'.format(step_num))

    with open(save_path, 'wb') as out_f:
        pickle.dump(checkpoint_state, out_f)

    logging.info('Checkpointed at {}'.format(save_path))


def load_checkpoint():
    """
    Load the generator, discriminator and their optimizer state dicts
    the path provided by FLAGS.load_checkpoint.
    """
    with open(FLAGS.load_checkpoint, 'rb') as in_f:
        checkpoint_state = pickle.load(in_f)
    logging.info('Loaded checkpoint {}'.format(FLAGS.load_checkpoint))
    return checkpoint_state


if __name__ == "__main__":
    app.run(train)
