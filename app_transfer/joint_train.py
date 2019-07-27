from os.path import join
import pickle

from absl import flags, app, logging
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from app_transfer.model.generator import Generator
from app_transfer.model.discriminator import Discriminator
from app_transfer.dataset import AsosDataset
from app_transfer.perceptual_loss_vgg import PerceptualLossVGG
from utils import (prepare_experiment_dirs,
                   get_tb_logger,
                   set_seeds)

FLAGS = flags.FLAGS

flags.DEFINE_float('generator_lr', 1e-5, "Starting learning rate of generator")
flags.DEFINE_float('discriminator_lr', 1e-4, "Starting learning rate of disciminator")
flags.DEFINE_string("generator_path", None, "Path to initial generator")
flags.DEFINE_string("discriminator_path", None, "Path to initial discriminator")


def train(unused_argv):
    """
    Trains a network to be able to generate images from img/pose
    pairs of asos data.
    """
    models_path = prepare_experiment_dirs()
    logger = get_tb_logger()
    set_seeds(131)

    generator = Generator()
    generator.load_state_dict(torch.load(FLAGS.generator_path))
    discriminator = Discriminator()
    discriminator.load_state_dict(torch.load(FLAGS.discriminator_path))
    perc_loss_vgg = PerceptualLossVGG()
    if FLAGS.use_cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        perc_loss_vgg = perc_loss_vgg.cuda()
    perc_loss_vgg.eval()

    dataset = AsosDataset(root_data_dir=FLAGS.data_dir,
                          overtrain=FLAGS.over_train)
    dataloader = DataLoader(dataset, batch_size=FLAGS.batch_size,
                            shuffle=True, num_workers=2, pin_memory=True)

    # TODO: Investigate these attn params
    g_optimizer = optim.Adam(generator.parameters(), lr=FLAGS.generator_lr, betas=(0., 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=FLAGS.discriminator_lr, betas=(0., 0.999))

    step_num = 0
    if FLAGS.load_checkpoint:
        checkpoint_state = load_checkpoint()
        generator.load_state_dict(checkpoint_state['generator'])
        discriminator.load_state_dict(checkpoint_state['discriminator'])
        g_optimizer.load_state_dict(checkpoint_state['g_optimizer'])
        d_optimizer.load_state_dict(checkpoint_state['d_optimizer'])
        step_num = checkpoint_state['step_num']

    # Placeholders
    d_loss = torch.Tensor([0.])

    for epoch in range(FLAGS.num_epochs):
        # Save a generator every 5 epochs
        if epoch % 5 == 0:
            save_path = join(models_path, '{}.pt'.format(epoch))
            torch.save(generator.state_dict(), save_path)

        for i, batch in enumerate(dataloader):
            batch = _prepare_batch_data(batch)
            app_img, app_pose_img, target_img, pose_img = batch

            # Train discriminator
            with torch.no_grad():
                gen_img = generator(app_img, app_pose_img, pose_img)

            # Hinge loss
            d_real_score = discriminator(app_img, app_pose_img, target_img, pose_img)
            d_fake_score = discriminator(app_img, app_pose_img, gen_img, pose_img)

            d_real_loss = nn.ReLU()(1. - d_real_score)
            d_fake_loss = nn.ReLU()(1. + d_fake_score)
            d_loss = d_fake_loss.mean() + d_real_loss.mean()

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train generator
            #gen_img = generator(app_img, app_pose_img, pose_img)

            #l1_loss = nn.L1Loss()(gen_img, target_img)
            #perceptual_loss = 0.3 * perc_loss_vgg(gen_img, target_img)
            if i % 2 == 0:
                gen_img = generator(app_img, app_pose_img, pose_img)
                # Only use GAN gradient ever 10 steps
                gan_loss = - 0.3 * discriminator(app_img, app_pose_img, gen_img, pose_img).mean()
                perceptual_loss = 0.3 * perc_loss_vgg(gen_img, target_img)
                l1_loss = nn.L1Loss()(gen_img, target_img)
                g_loss = l1_loss + gan_loss + perceptual_loss
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()
                #g_loss = l1_loss + gan_loss + perceptual_loss
            #else:
            #    g_loss = l1_loss + perceptual_loss

            #g_optimizer.zero_grad()
            #g_loss.backward()
            #g_optimizer.step()

            if step_num % FLAGS.tb_log_interval == 0:
                log_results(epoch, step_num, logger, gen_img,
                            d_loss, g_loss, l1_loss, gan_loss, perceptual_loss)

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
    app_pose_img = batch['app_pose_img']
    target_img = batch['target_img']
    pose_img = batch['pose_img']

    if FLAGS.use_cuda:
        app_img = app_img.cuda()
        app_pose_img = app_pose_img.cuda()
        target_img = target_img.cuda()
        pose_img = pose_img.cuda()

    app_img = nn.MaxPool2d(kernel_size=4)(app_img)
    app_pose_img = nn.MaxPool2d(kernel_size=4)(app_pose_img)
    target_img = nn.MaxPool2d(kernel_size=4)(target_img)
    pose_img = nn.MaxPool2d(kernel_size=4)(pose_img)

    return app_img, app_pose_img, target_img, pose_img


def log_results(epoch, step_num, writer, gen_img, d_loss,
                g_loss, l1_loss, gan_loss, perceptual_loss):
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
    writer.add_scalar('Train/perceptual_loss', perceptual_loss, step_num)


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
