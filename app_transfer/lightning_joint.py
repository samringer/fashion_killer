from os.path import join
from pathlib import Path
import pickle

from absl import flags, app, logging
import argparse
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, DistributedSampler
import pytorch_lightning as pl
from test_tube import Experiment

from app_transfer.model.generator import Generator
from app_transfer.model.discriminator import Discriminator
from app_transfer.dataset import AsosDataset
from app_transfer.model.perceptual_loss_vgg import PerceptualLossVGG
from utils import (prepare_experiment_dirs,
                   get_tb_logger,
                   set_seeds)


class MyLightning(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.generator = torch.load(args.gen_path)
        self.discriminator = torch.load(args.disc_path)
        self.perc_loss_vgg = PerceptualLossVGG()

        self.args = args
        self.save_dir = Path(args.task_path)/args.exp_name/'models'
        self.save_dir.mkdir(exist_ok=True)

    def forward(self, app_img, app_pose_img, pose_img):
        return self.generator(app_img, app_pose_img, pose_img)

    def training_step(self, batch, batch_nb, optimizer_idx):
        app_img = batch['app_img']
        app_pose_img = batch['app_pose_img']
        target_img = batch['target_img']
        pose_img = batch['pose_img']

        # TODO: Add in second step
        # Train generator every second step
        if optimizer_idx == 0:# and self.global_step % 2 == 0:
            gen_img = self.generator(app_img, app_pose_img, pose_img)
            gan_loss = - 0.02 * self.discriminator(app_img,
                                                   app_pose_img,
                                                   pose_img,
                                                   gen_img).mean()
            perceptual_loss = 0.02 * self.perc_loss_vgg(gen_img,
                                                        target_img)
            l1_loss = nn.L1Loss()(gen_img, target_img)
            g_loss = l1_loss + gan_loss + perceptual_loss

            if self.global_step % self.args.tb_log_interval == 0:
                self.log_gen_results(gen_img, g_loss, l1_loss,
                                     gan_loss, perceptual_loss)
            return {'loss': g_loss}

        # Train discriminator
        if optimizer_idx == 1:
            with torch.no_grad():
                gen_img = self.generator(app_img, app_pose_img,
                                         pose_img)

            # Hinge loss
            d_real_score = self.discriminator(app_img, app_pose_img,
                                              pose_img, target_img)
            d_fake_score = self.discriminator(app_img, app_pose_img,
                                              pose_img, gen_img)

            d_real_loss = nn.ReLU()(1. - d_real_score)
            d_fake_loss = nn.ReLU()(1. + d_fake_score)
            d_loss = d_fake_loss.mean() + d_real_loss.mean()

            if self.global_step % self.args.tb_log_interval == 0:
                self.experiment.add_scalar('Train/d_loss',
                                           d_loss.item(),
                                           self.global_step)
            return {'loss': d_loss}

    def validation_step(self, batch, batch_nb):
        app_img = batch['app_img']
        app_pose_img = batch['app_pose_img']
        target_img = batch['target_img']
        pose_img = batch['pose_img']

        gen_img = self.generator.forward(app_img, app_pose_img,
                                         pose_img)
        gan_loss = - 0.02 * self.discriminator(app_img,
                                               app_pose_img,
                                               pose_img,
                                               gen_img).mean()
        perceptual_loss = 0.02 * self.perc_loss_vgg(gen_img,
                                                    target_img)
        l1_loss = nn.L1Loss()(gen_img, target_img)
        g_loss = l1_loss + gan_loss + perceptual_loss
        return {'val_loss': g_loss}

    def validation_end(self, outputs):
        val_loss_mean = 0
        for output in outputs:
            val_loss_mean += output['val_loss']
        val_loss_mean /= len(outputs)
        self.experiment.add_scalar('Val/g_loss',
                                   val_loss_mean,
                                   self.global_step)
        return {'val_loss': val_loss_mean.item()}

    def on_save_checkpoint(self, checkpoint):
        if self.current_epoch % 5 == 0:
            save_path = (self.save_dir/str(self.current_epoch)).with_suffix('.pt')
            torch.save(self.generator, save_path.as_posix())

    def configure_optimizers(self):
        g_opt = optim.Adam(self.generator.parameters(),
                           lr=self.args.gen_lr,
                           betas=(0., 0.999))
        d_opt = optim.Adam(self.discriminator.parameters(),
                           lr=self.args.disc_lr,
                           betas=(0., 0.999))
        g_lr_scheduler = StepLR(g_opt, step_size=40, gamma=0.8)
        d_lr_scheduler = StepLR(d_opt, step_size=40, gamma=0.8)
        return [g_opt, d_opt], [g_lr_scheduler, d_lr_scheduler]

    @pl.data_loader
    def tng_dataloader(self):
        dataset = AsosDataset(root_data_dir=self.args.train_data_dir,
                              overtrain=self.args.overtrain)
        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset,
                                batch_size=self.args.batch_size,
                                sampler=sampler,
                                num_workers=5, pin_memory=True)
        return dataloader

    @pl.data_loader
    def val_dataloader(self):
        dataset = AsosDataset(root_data_dir=self.args.val_data_dir,
                              overtrain=self.args.overtrain)
        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset,
                                batch_size=self.args.batch_size,
                                sampler=sampler,
                                num_workers=5, pin_memory=True)
        return dataloader

    def log_gen_results(self, gen_img, g_loss, l1_loss, gan_loss,
                        perceptual_loss):
        """
        Log the results using tensorboardx so they can be
        viewed using a tensorboard server.
        """
        gen_img = gen_img[0].detach().cpu()
        img_file_name = 'generated_img/{}'.format(self.current_epoch)
        self.experiment.add_image(img_file_name, gen_img,
                                  self.global_step)
        self.experiment.add_scalar('Train/g_loss', g_loss.item(),
                                   self.global_step)
        self.experiment.add_scalar('Train/l1_loss', l1_loss.item(),
                                   self.global_step)
        self.experiment.add_scalar('Train/gan_loss', gan_loss.item(),
                                   self.global_step)
        self.experiment.add_scalar('Train/perceptual_loss',
                                   perceptual_loss.item(),
                                   self.global_step)


def train(args):
    set_seeds()
    model = MyLightning(args)
    exp = Experiment(name=args.exp_name, save_dir=args.task_path)
    exp.tag({
        'batch_size': args.batch_size,
        'gen_lr': args.gen_lr,
        'disc_lr': args.disc_lr,
        'description': 'full size img training'
    })

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=join(args.task_path, args.exp_name, 'weights'),
        save_best_only=False,
        verbose=1,
        period=1
    )

    trainer = pl.Trainer(max_nb_epochs=args.num_epochs,
                         distributed_backend='ddp',
                         gpus=[0, 1],
                         experiment=exp,
                         accumulate_grad_batches=4,
                         amp_level='O1',
                         use_amp=True,
                         checkpoint_callback=checkpoint_callback)
    trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('task_path', type=str)
    parser.add_argument('exp_name', type=str)
    parser.add_argument('train_data_dir', type=str)
    parser.add_argument('val_data_dir', type=str)
    parser.add_argument('gen_path', type=str)
    parser.add_argument('disc_path', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--tb_log_interval', default=30, type=int)
    parser.add_argument('--gen_lr', default=4e-4, type=float)
    parser.add_argument('--disc_lr', default=4e-4, type=float)
    # TODO: Lightning should be able to do this
    parser.add_argument('--overtrain', action='store_true')
    train_args = parser.parse_args()
    train(train_args)
