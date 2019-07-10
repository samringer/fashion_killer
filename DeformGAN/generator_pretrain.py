from os.path import join
import pickle

from absl import flags, app, logging
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchsummary import summary
from apex import amp

from DeformGAN.model.generator import Generator
from DeformGAN.dataset import AsosDataset
from DeformGAN.perceptual_loss_vgg import PerceptualLossVGG
from DeformGAN.init_generator_from_previous import init_generator_from_previous
from utils import (prepare_experiment_dirs,
                   get_tb_logger,
                   set_seeds)


FLAGS = flags.FLAGS
flags.DEFINE_string('gen_init_path', None, "Path to initialise generator from")


def train(unused_argv):
    """
    Trains a network to be able to generate images from img/pose
    pairs of asos data.
    """
    models_path = prepare_experiment_dirs()
    logger = get_tb_logger()
    set_seeds(257)

    if FLAGS.gen_init_path:
        generator = init_generator_from_previous(FLAGS.gen_init_path)
    else:
        generator = Generator()
    perceptual_loss_vgg = PerceptualLossVGG()
    if FLAGS.use_cuda:
        generator = generator.cuda()
        perceptual_loss_vgg = perceptual_loss_vgg.cuda()
    perceptual_loss_vgg.eval()

    summary(generator, input_size=[(3, 256, 256) for _ in range(3)])
    set_seeds(257)  # Doing a summary skrews the rng so reset seeds

    dataset = AsosDataset(root_data_dir=FLAGS.data_dir,
                          overtrain=FLAGS.over_train)
    dataloader = DataLoader(dataset, batch_size=FLAGS.batch_size,
                            shuffle=True, num_workers=3, pin_memory=True)

    g_optimizer = optim.Adam(generator.parameters(),
                             lr=FLAGS.learning_rate, betas=(0.5, 0.999))

    lr_scheduler = StepLR(g_optimizer, step_size=75, gamma=0.5)

    step_num = 0
    if FLAGS.load_checkpoint:
        checkpoint_state = load_checkpoint()
        generator.load_state_dict(checkpoint_state['generator'])
        g_optimizer.load_state_dict(checkpoint_state['g_optimizer'])
        lr_scheduler.load_state_dict(checkpoint_state['lr_scheduler'])
        step_num = checkpoint_state['step_num']

    for epoch in range(FLAGS.num_epochs):

        lr_scheduler.step()

        if epoch % 5 == 0:
            save_path = join(models_path, '{}.pt'.format(epoch))
            torch.save(generator.state_dict(), save_path)

        for batch in dataloader:
            app_img = batch['app_img']
            app_pose_img = batch['app_pose_img']
            target_img = batch['target_img']
            pose_img = batch['pose_img']

            if FLAGS.use_cuda:
                app_img = app_img.cuda()
                app_pose_img = app_pose_img.cuda()
                target_img = target_img.cuda()
                pose_img = pose_img.cuda()

            #app_img = nn.MaxPool2d(kernel_size=4)(app_img)
            #app_pose_img = nn.MaxPool2d(kernel_size=4)(app_pose_img)
            #target_img = nn.MaxPool2d(kernel_size=4)(target_img)
            #pose_img = nn.MaxPool2d(kernel_size=4)(pose_img)

            gen_img = generator(app_img, app_pose_img, pose_img)
            l1_loss = nn.L1Loss()(gen_img, target_img)
            perceptual_loss = 0.3 * perceptual_loss_vgg(gen_img, target_img)
            loss = l1_loss + perceptual_loss
            loss.backward()

            if step_num % 4 == 0:
                # Trick to increase the effective batch size
                # By a factor of 4
                clip_grad_norm_(generator.parameters(), 5)
                g_optimizer.step()
                g_optimizer.zero_grad()

            if step_num % FLAGS.tb_log_interval == 0:
                log_results(epoch, step_num, logger, gen_img, loss, l1_loss, perceptual_loss)

            if step_num % FLAGS.checkpoint_interval == 0:
                # TODO: Add in lr scheduler
                checkpoint_state = {
                    'generator': generator.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'step_num': step_num,
                    'lr_scheduler': lr_scheduler.state_dict()
                }
                save_checkpoint(checkpoint_state)
            logging.info(f"{step_num} {loss.item():.4f}")
            step_num += 1


def log_results(epoch, step_num, writer, gen_img, loss, l1_loss,
                perceptual_loss):
    """
    Log the results using tensorboardx so they can be
    viewed using a tensorboard server.
    """
    gen_img = gen_img[0].detach().cpu()
    img_file_name = 'generated_img/{}'.format(epoch)
    writer.add_image(img_file_name, gen_img, step_num)
    writer.add_scalar('Train/total_loss', loss.item(), step_num)
    writer.add_scalar('Train/l1_loss', l1_loss.item(), step_num)
    writer.add_scalar('Train/perceptual_loss', perceptual_loss.item(), step_num)


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
    Load the generator its optimizer state dict at
    the path provided by FLAGS.load_checkpoint.
    """
    with open(FLAGS.load_checkpoint, 'rb') as in_f:
        checkpoint_state = pickle.load(in_f)
    logging.info('Loaded checkpoint {}'.format(FLAGS.load_checkpoint))
    return checkpoint_state


if __name__ == "__main__":
    app.run(train)
