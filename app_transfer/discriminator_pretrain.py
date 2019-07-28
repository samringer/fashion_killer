import pickle
from os.path import join

from absl import flags, app, logging
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from app_transfer.model.discriminator import Discriminator
from app_transfer.dataset import AsosDataset
from utils import (prepare_experiment_dirs,
                   get_tb_logger,
                   set_seeds,
                   device)


FLAGS = flags.FLAGS
flags.DEFINE_string("generator_path", None, "Which generator to train discriminator with")


def train(unused_argv):
    """
    Pretrains a discriminator using a pretrained generator
    before joint training.
    """
    models_path = prepare_experiment_dirs()
    logger = get_tb_logger()
    set_seeds(128)

    generator = torch.load(FLAGS.generator_path).to(device)
    discriminator = Discriminator().to(device)

    dataset = AsosDataset(root_data_dir=FLAGS.data_dir,
                          overtrain=FLAGS.over_train)
    dataloader = DataLoader(dataset, batch_size=FLAGS.batch_size,
                            shuffle=True, num_workers=2, pin_memory=True)

    d_optimizer = optim.Adam(discriminator.parameters(),
                             lr=FLAGS.learning_rate, betas=(0.5, 0.999))

    lr_scheduler = StepLR(d_optimizer, step_size=100, gamma=0.5)

    step_num = 0
    if FLAGS.load_checkpoint:
        checkpoint_state = load_checkpoint()
        discriminator.load_state_dict(checkpoint_state['discriminator'])
        d_optimizer.load_state_dict(checkpoint_state['d_optimizer'])
        step_num = checkpoint_state['step_num']

    for epoch in range(FLAGS.num_epochs):

        lr_scheduler.step()

        for batch in dataloader:
            if step_num >= 800:
                save_path = join(models_path, 'final.pt')
                torch.save(discriminator, save_path)
                return

            app_img = batch['app_img'].to(device)
            app_pose_img = batch['app_pose_img'].to(device)
            target_img = batch['target_img'].to(device)
            pose_img = batch['pose_img'].to(device)

            app_img = nn.MaxPool2d(kernel_size=4)(app_img)
            app_pose_img = nn.MaxPool2d(kernel_size=4)(app_pose_img)
            target_img = nn.MaxPool2d(kernel_size=4)(target_img)
            pose_img = nn.MaxPool2d(kernel_size=4)(pose_img)

            with torch.no_grad():
                gen_img = generator(app_img, app_pose_img, pose_img)

            d_optimizer.zero_grad()
            real_score = discriminator(app_img, app_pose_img, target_img,
                                       pose_img)
            gen_score = discriminator(app_img, app_pose_img, gen_img,
                                      pose_img)
            real_loss = (nn.ReLU()(1. - real_score)).mean()
            gen_loss = (nn.ReLU()(1. + gen_score)).mean()
            loss = real_loss + gen_loss
            loss.backward()
            d_optimizer.step()

            if step_num % FLAGS.tb_log_interval == 0:
                log_results(step_num, logger, loss)

            if step_num % FLAGS.checkpoint_interval == 0:
                # TODO: Add in lr scheduler
                checkpoint_state = {
                    'discriminator': discriminator.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    'step_num': step_num
                }
                save_checkpoint(checkpoint_state)
            logging.info(f"{step_num} {loss.item():.4f}")
            step_num += 1


def log_results(step_num, writer, loss):
    """
    Log the results using tensorboardx so they can be
    viewed using a tensorboard server.
    """
    writer.add_scalar('Train/total_loss', loss.item(), step_num)


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
