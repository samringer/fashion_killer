import os
import sys
from os.path import join
from time import sleep
import pickle

import argparse
import torch
from absl import logging
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import torch.distributed as dist
from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from torchsummary import summary

from app_transfer.model.generator import Generator
from app_transfer.dataset import AsosDataset
from app_transfer.model.perceptual_loss_vgg import PerceptualLossVGG
from utils import (prepare_experiment_dirs,
                   get_tb_logger,
                   set_seeds,
                   device)


def train(args):
    """
    Trains a network to be able to generate images from img/pose
    pairs of asos data.
    """

    if args.local_rank == 0:
        models_path = prepare_experiment_dirs(args.task_path,
                                              args.exp_name)
        tb_logger = get_tb_logger(args.task_path, args.exp_name)

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')

    set_seeds(257)
    generator = Generator().cuda()
    perceptual_loss_vgg = PerceptualLossVGG().cuda()

    if args.local_rank == 0:
        dummy_inp_sz = [(3, 256, 256), (21, 256, 256), (21, 256, 256)]
        summary(generator, input_size=dummy_inp_sz)
        # Doing a summary skrews the rng so reset seedsummy_input_size)
        set_seeds(257)

    g_optimizer = optim.Adam(generator.parameters(),
                             lr=args.learning_rate, betas=(0.5, 0.999))
    if args.use_fp16:
        [generator, perceptual_loss_vgg], g_optimizer = amp.initialize([generator, perceptual_loss_vgg], g_optimizer, opt_level='O1')

    if args.distributed:
        generator = DDP(generator)
        perceptual_loss_vgg = DDP(perceptual_loss_vgg)
        set_seeds(257 + args.local_rank)


    dataset = AsosDataset(root_data_dir=args.data_dir,
                          overtrain=args.overtrain)

    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=5, pin_memory=True)

    lr_scheduler = StepLR(g_optimizer, step_size=40, gamma=0.8)

    step_num = 0
    # TODO: Work out how to get this back in
    #if FLAGS.load_checkpoint:
    #    checkpoint_state = load_checkpoint()
    #    generator.load_state_dict(checkpoint_state['generator'])
    #    g_optimizer.load_state_dict(checkpoint_state['g_optimizer'])
    #    lr_scheduler.load_state_dict(checkpoint_state['lr_scheduler'])
    #    step_num = checkpoint_state['step_num']

    for epoch in range(args.num_epochs):
        if epoch % 2 == 0 and args.local_rank == 0:
            save_path = join(models_path, '{}.pt'.format(epoch))
            torch.save(generator.state_dict(), save_path)

        for batch in dataloader:
            # TODO: Change these cudas to the device thing
            app_img = batch['app_img'].cuda()
            app_pose_img = batch['app_pose_img'].cuda()
            target_img = batch['target_img'].cuda()
            pose_img = batch['pose_img'].cuda()

            #app_img = nn.MaxPool2d(kernel_size=2)(app_img)
            #app_pose_img = nn.MaxPool2d(kernel_size=2)(app_pose_img)
            #target_img = nn.MaxPool2d(kernel_size=2)(target_img)
            #pose_img = nn.#MaxPool2d(kernel_size=2)(pose_img)

            gen_img = generator(app_img, app_pose_img, pose_img)
            l1_loss = nn.L1Loss()(gen_img, target_img)
            perceptual_loss = 0.02 * perceptual_loss_vgg(gen_img, target_img)
            loss = l1_loss + perceptual_loss
            if args.use_fp16:
                with amp.scale_loss(loss, g_optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            clip_grad_norm_(generator.parameters(), 5)
            g_optimizer.step()
            g_optimizer.zero_grad()

            if step_num % args.tb_log_interval == 0 \
               and args.local_rank == 0:
                tb_log_results(epoch, step_num, tb_logger, gen_img, loss, l1_loss, perceptual_loss)

            if step_num % args.checkpoint_interval == 0 \
               and args.local_rank == 0:
                sleep(5)
                checkpoint_state = {
                    'generator': generator.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'step_num': step_num,
                    'lr_scheduler': lr_scheduler.state_dict()
                }
                sleep(5)
                save_checkpoint(checkpoint_state, args.task_path,
                                args.exp_name)
            print(f"{step_num} {loss.item():.4f} {args.local_rank}")
            step_num += 1
        lr_scheduler.step()


def tb_log_results(epoch, step_num, writer, gen_img, loss, l1_loss,
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


def save_checkpoint(checkpoint_state, task_path, exp_name):
    """
    Checkpoint the train_state.
    Saves them all together in a tuple representing the train state.
    """
    exp_dir = join(task_path, exp_name)
    step_num = checkpoint_state['step_num']
    save_path = join(exp_dir, 'models', '{}.chk'.format(step_num))

    with open(save_path, 'wb') as out_f:
        pickle.dump(checkpoint_state, out_f)

    print('Checkpointed at {}'.format(save_path))


def load_checkpoint(chk_path):
    """
    Load the generator its optimizer state dict at
    the path provided by FLAGS.load_checkpoint.
    """
    with open(chk_path, 'rb') as in_f:
        checkpoint_state = pickle.load(in_f)
    print('Loaded checkpoint {}'.format(chk_path))
    return checkpoint_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_path', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--data_dir',
                        default='/home/sam/data/asos/1307_clean/train')
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--checkpoint_interval', type=int,
                        default=20000)
    parser.add_argument('--tb_log_interval', type=int, default=30)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--use_fp16', action='store_true')
    parser.add_argument('--overtrain', action='store_true')
    train_args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    train(train_args)
