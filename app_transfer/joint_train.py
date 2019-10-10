from os.path import join
import argparse
import pickle
import time

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import torch.distributed as dist
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

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
        last_step_time = time.time()
        models_path = prepare_experiment_dirs(args.task_path,
                                              args.exp_name)
        tb_logger = get_tb_logger(args.task_path, args.exp_name)

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')

    generator = torch.load(args.gen_path)
    generator = generator.cuda()

    discriminator = torch.load(args.disc_path)
    discriminator = discriminator.cuda()

    perceptual_loss_vgg = PerceptualLossVGG().cuda()
    #perceptual_loss_vgg.eval()

    dataset = AsosDataset(root_data_dir=args.data_dir,
                          overtrain=args.overtrain)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=5, pin_memory=True)

    g_optimizer = optim.Adam(generator.parameters(),
                             lr=args.gen_lr, betas=(0., 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(),
                             lr=args.disc_lr, betas=(0., 0.999))

    g_lr_scheduler = StepLR(g_optimizer, step_size=50, gamma=0.8)
    d_lr_scheduler = StepLR(d_optimizer, step_size=50, gamma=0.8)

    if args.use_fp16:
        [generator, discriminator, perceptual_loss_vgg], \
        [g_optimizer, d_optimizer] = amp.initialize([generator,
                                                     discriminator,
                                                     perceptual_loss_vgg],
                                                    [g_optimizer,
                                                     d_optimizer],
                                                    opt_level='O1')

    if args.distributed:
        generator = DDP(generator, delay_allreduce=True)
        discriminator = DDP(discriminator, delay_allreduce=True)
        perceptual_loss_vgg = DDP(perceptual_loss_vgg, delay_allreduce=True)
        set_seeds(257 + args.local_rank)


    step_num = 0
    if args.load_checkpoint:
        chk_state = load_checkpoint(args)
        generator.load_state_dict(chk_state['generator'])
        discriminator.load_state_dict(chk_state['discriminator'])
        g_optimizer.load_state_dict(chk_state['g_optimizer'])
        d_optimizer.load_state_dict(chk_state['d_optimizer'])
        g_lr_scheduler.load_state_dict(chk_state['g_lr_scheduler'])
        d_lr_scheduler.load_state_dict(chk_state['d_lr_scheduler'])
        step_num = chk_state['step_num']

    for epoch in range(args.num_epochs):


        # Save a generator every 5 epochs
        #if epoch % 5 == 0:
        #    save_path = join(models_path, '{}.pt'.format(epoch))
        #    torch.save(generator.state_dict(), save_path)

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
            if args.use_fp16:
                with amp.scale_loss(d_loss, d_optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                d_loss.backward()
            clip_grad_norm_(discriminator.parameters(), 1)
            d_optimizer.step()

            # Train generator every second step
            if i % 2 == 0:
                gen_img = generator(app_img, app_pose_img, pose_img)
                gan_loss = - 0.02 * discriminator(app_img,
                                                  app_pose_img,
                                                  pose_img,
                                                  gen_img).mean()
                perceptual_loss = 0.02 * perceptual_loss_vgg(gen_img,
                                                             target_img)
                l1_loss = nn.L1Loss()(gen_img, target_img)
                #h_loss = 10 * discriminator.hierachy_loss(app_img,
                #                                          app_pose_img,
                #                                          pose_img,
                #                                          target_img,
                #                                          gen_img)

                g_loss = l1_loss + gan_loss + perceptual_loss #+ h_loss

                g_optimizer.zero_grad()
                if args.use_fp16:
                    with amp.scale_loss(g_loss, g_optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    g_loss.backward()
                clip_grad_norm_(generator.parameters(), 1)
                g_optimizer.step()

            if step_num % args.tb_log_interval == 0 \
               and args.local_rank == 0:
                log_results(epoch, step_num, tb_logger, gen_img,
                            d_loss, g_loss, l1_loss, gan_loss,
                            perceptual_loss) #, h_loss)

            if step_num % args.checkpoint_interval == 0 \
               and args.local_rank == 0:
                checkpoint_state = {
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    'step_num': step_num,
                    'g_lr_scheduler': g_lr_scheduler.state_dict(),
                    'd_lr_scheduler': d_lr_scheduler.state_dict()
                }
                save_checkpoint(checkpoint_state, args)

            if args.local_rank == 0:
                step_time = time.time() - last_step_time
                last_step_time = time.time()
                print(f"{step_num} d_loss: {d_loss.item():.4f} g_loss: {g_loss.item():.4f} in {step_time:.2f}s")

            step_num += 1

        # Step at the end of each epoch
        g_lr_scheduler.step()
        d_lr_scheduler.step()


def _prepare_batch_data(batch):
    app_img = batch['app_img'].to(device)
    app_pose_img = batch['app_pose_img'].to(device)
    target_img = batch['target_img'].to(device)
    pose_img = batch['pose_img'].to(device)

    #app_img = nn.MaxPool2d(kernel_size=2)(app_img)
    #app_pose_img = nn.MaxPool2d(kernel_size=2)(app_pose_img)
    #target_img = nn.MaxPool2d(kernel_size=2)(target_img)
    #pose_img = nn.MaxPool2d(kernel_size=2)(pose_img)

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
    writer.add_scalar('Train/g_loss', g_loss.item(), step_num)
    writer.add_scalar('Train/l1_loss', l1_loss.item(), step_num)
    writer.add_scalar('Train/gan_loss', gan_loss.item(), step_num)
    writer.add_scalar('Train/d_loss', d_loss.item(), step_num)
    writer.add_scalar('Train/perceptual_loss', perceptual_loss.item(), step_num)
    #writer.add_scalar('Train/hierachy_loss', h_loss, step_num)


def save_checkpoint(checkpoint_state, args):
    """
    Checkpoint the train_state.
    Saves them all together in a tuple representing the train state.
    """
    exp_dir = join(args.task_path, args.exp_name)
    step_num = checkpoint_state['step_num']
    save_path = join(exp_dir, 'models', '{}.chk'.format(step_num))

    with open(save_path, 'wb') as out_f:
        pickle.dump(checkpoint_state, out_f)

    print('Checkpointed at {}'.format(save_path))


def load_checkpoint(args):
    """
    Load the generator, discriminator and their optimizer state dicts
    the path provided by FLAGS.load_checkpoint.
    """
    with open(args.load_checkpoint, 'rb') as in_f:
        checkpoint_state = pickle.load(in_f)
    print('Loaded checkpoint {}'.format(args.load_checkpoint))
    return checkpoint_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('task_path', type=str)
    parser.add_argument('exp_name', type=str)
    parser.add_argument('gen_path', type=str)
    parser.add_argument('disc_path', type=str)
    parser.add_argument('--data_dir',
                        default='/home/sam/data/asos/1307_clean/train')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gen_lr', type=float, default=1e-4)
    parser.add_argument('--disc_lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--checkpoint_interval', type=int,
                        default=20000)
    parser.add_argument('--tb_log_interval', type=int, default=30)
    parser.add_argument('--load_checkpoint', type=str, default=None)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--use_fp16', action='store_true')
    parser.add_argument('--overtrain', action='store_true')
    train_args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    train(train_args)
