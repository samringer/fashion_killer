from os.path import join
import pickle

from absl import flags, app, logging
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from app_transfer.model.generator import Generator
from app_transfer.model.discriminator import Discriminator
from app_transfer.dataset import AsosDataset
from app_transfer.perceptual_loss_vgg import PerceptualLossVGG
from utils import (prepare_experiment_dirs,
                   get_tb_logger,
                   set_seeds,
                   device)

FLAGS = flags.FLAGS

flags.DEFINE_float('gen_lr', 1e-5, "Start lr of generator")
flags.DEFINE_float('disc_lr', 1e-4, "Start lr of disciminator")
flags.DEFINE_string("gen_path", None, "Path to initial generator")
flags.DEFINE_string("disc_path", None, "Path to initial discriminator")


def train(unused_argv):
    """
    Trains a network to be able to generate images from img/pose
    pairs of asos data.
    """
    models_path = prepare_experiment_dirs()
    logger = get_tb_logger()
    set_seeds(131)


    generator = Generator()
    generator.load_state_dict(torch.load(FLAGS.gen_path))
    generator = generator.to(device)

    discriminator = Discriminator()
    discriminator.load_state_dict(torch.load(FLAGS.disc_path))
    discriminator = discriminator.to(device)

    perc_loss_vgg = PerceptualLossVGG().to(device)
    perc_loss_vgg.eval()

    dataset = AsosDataset(root_data_dir=FLAGS.data_dir,
                          overtrain=FLAGS.over_train)
    dataloader = DataLoader(dataset, batch_size=FLAGS.batch_size,
                            shuffle=True, num_workers=5, pin_memory=True)

    g_optimizer = optim.Adam(generator.parameters(),
                             lr=FLAGS.gen_lr, betas=(0., 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(),
                             lr=FLAGS.disc_lr, betas=(0., 0.999))

    g_lr_scheduler = StepLR(g_optimizer, step_size=20, gamma=0.8)
    d_lr_scheduler = StepLR(d_optimizer, step_size=20, gamma=0.8)

    step_num = 0
    if FLAGS.load_checkpoint:
        chk_state = load_checkpoint()
        generator.load_state_dict(chk_state['generator'])
        discriminator.load_state_dict(chk_state['discriminator'])
        g_optimizer.load_state_dict(chk_state['g_optimizer'])
        d_optimizer.load_state_dict(chk_state['d_optimizer'])
        g_lr_scheduler.load_state_dict(chk_state['g_lr_scheduler'])
        d_lr_scheduler.load_state_dict(chk_state['d_lr_scheduler'])
        step_num = chk_state['step_num']

    for epoch in range(FLAGS.num_epochs):

        g_lr_scheduler.step()
        d_lr_scheduler.step()

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
            d_real_score = discriminator(app_img, app_pose_img,
                                         pose_img, target_img)
            d_fake_score = discriminator(app_img, app_pose_img,
                                         pose_img, gen_img)

            d_real_loss = nn.ReLU()(1. - d_real_score)
            d_fake_loss = nn.ReLU()(1. + d_fake_score)
            d_loss = d_fake_loss.mean() + d_real_loss.mean()

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train generator every second step
            if i % 2 == 0:
                gen_img = generator(app_img, app_pose_img, pose_img)
                gan_loss = - 0.02 * discriminator(app_img,
                                                  app_pose_img,
                                                  pose_img,
                                                  gen_img).mean()
                perceptual_loss = 0.02 * perc_loss_vgg(gen_img,
                                                       target_img)
                l1_loss = nn.L1Loss()(gen_img, target_img)
                #h_loss = 10 * discriminator.hierachy_loss(app_img,
                #                                          app_pose_img,
                #                                          pose_img,
                #                                          target_img,
                #                                          gen_img)

                g_loss = l1_loss + gan_loss + perceptual_loss #+ h_loss

                g_optimizer.zero_grad()
                g_loss.backward()
                clip_grad_norm_(generator.parameters(), 5)
                g_optimizer.step()

            if step_num % FLAGS.tb_log_interval == 0:
                log_results(epoch, step_num, logger, gen_img,
                            d_loss, g_loss, l1_loss, gan_loss,
                            perceptual_loss) #, h_loss)

            if step_num % FLAGS.checkpoint_interval == 0:
                checkpoint_state = {
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    'step_num': step_num,
                    'g_lr_scheduler': g_lr_scheduler.state_dict(),
                    'd_lr_scheduler': d_lr_scheduler.state_dict()
                }
                save_checkpoint(checkpoint_state)
            logging.info(f"{step_num} {d_loss.item():.4f} {g_loss.item():.4f}")
            step_num += 1


def _prepare_batch_data(batch):
    app_img = batch['app_img'].to(device)
    app_pose_img = batch['app_pose_img'].to(device)
    target_img = batch['target_img'].to(device)
    pose_img = batch['pose_img'].to(device)

    app_img = nn.MaxPool2d(kernel_size=2)(app_img)
    app_pose_img = nn.MaxPool2d(kernel_size=2)(app_pose_img)
    target_img = nn.MaxPool2d(kernel_size=2)(target_img)
    pose_img = nn.MaxPool2d(kernel_size=2)(pose_img)

    return app_img, app_pose_img, target_img, pose_img


def log_results(epoch, step_num, writer, gen_img, d_loss,
                g_loss, l1_loss, gan_loss, perceptual_loss): #, h_loss):
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
    #writer.add_scalar('Train/hierachy_loss', h_loss, step_num)


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
