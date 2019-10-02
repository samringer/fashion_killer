from os.path import join
from pathlib import Path
import pickle

from absl import flags, app, logging
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, DistributedSampler
import pytorch_lightning as pl
from test_tube import Experiment

from app_transfer.model.generator import Generator
from app_transfer.dataset import AsosDataset
from app_transfer.model.perceptual_loss_vgg import PerceptualLossVGG
from utils import (prepare_experiment_dirs,
                   get_tb_logger,
                   set_seeds)


class MyLightning(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.generator = Generator()
        self.perceptual_loss_vgg = PerceptualLossVGG()
        self.args = args
        self.save_dir = Path(args.task_path)/args.exp_name/'weights'
        self.save_dir.mkdir(exist_ok=True)

    def forward(self, app_img, app_pose_img, pose_img):
        return self.generator(app_img, app_pose_img, pose_img)

    def training_step(self, batch, batch_nb):
        app_img = batch['app_img']
        app_pose_img = batch['app_pose_img']
        target_img = batch['target_img']
        pose_img = batch['pose_img']

        gen_img = self.forward(app_img, app_pose_img, pose_img)

        l1_loss = nn.L1Loss()(gen_img, target_img)
        perceptual_loss = 0.02 * self.perceptual_loss_vgg(gen_img,
                                                          target_img)
        loss = l1_loss + perceptual_loss

        if self.global_step % self.args.tb_log_interval == 0:
            self.log_results(gen_img, loss, l1_loss, perceptual_loss)

        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        app_img = batch['app_img']
        app_pose_img = batch['app_pose_img']
        target_img = batch['target_img']
        pose_img = batch['pose_img']

        gen_img = self.forward(app_img, app_pose_img, pose_img)

        l1_loss = nn.L1Loss()(gen_img, target_img)
        perceptual_loss = 0.02 * self.perceptual_loss_vgg(gen_img,
                                                          target_img)
        loss = l1_loss + perceptual_loss
        return {'val_loss': loss}

    def on_save_checkpoint(self, checkpoint):
        save_path = (self.save_dir/str(self.current_epoch)).with_suffix('.pt')
        torch.save(self.generator, save_path.as_posix())

    def configure_optimizers(self):
        return optim.Adam(self.generator.parameters(),
                          lr=self.args.learning_rate,
                          betas=(0.5, 0.999))

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


    def log_results(self, gen_img, loss, l1_loss, perceptual_loss):
        """
        Log the results using tensorboardx so they can be
        viewed using a tensorboard server.
        """
        gen_img = gen_img[0].detach().cpu()
        img_file_name = 'generated_img/{}'.format(self.current_epoch)
        self.experiment.add_image(img_file_name, gen_img,
                                  self.global_step)
        self.experiment.add_scalar('Train/total_loss', loss.item(),
                                   self.global_step)
        self.experiment.add_scalar('Train/l1_loss', l1_loss.item(),
                                   self.global_step)
        self.experiment.add_scalar('Train/perceptual_loss',
                                   perceptual_loss.item(), self.global_step)


def train(args):
    set_seeds()
    model = MyLightning(args)
    exp = Experiment(name=args.exp_name, save_dir=args.task_path)
    exp.tag({
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'description': 'full size img training'
    })

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=join(args.task_path, args.exp_name, 'weights'),
        save_best_only=False,
        verbose=1,
        period=1
    )

    trainer = pl.Trainer(max_nb_epochs=3,
                         distributed_backend='ddp',
                         gpus=[0, 1],
                         experiment=exp,
                         accumulate_grad_batches=2,
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
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--tb_log_interval', default=30, type=int)
    parser.add_argument('--learning_rate', default=4e-4, type=float)
    # TODO: Lightning should be able to do this
    parser.add_argument('--overtrain', action='store_true')
    train_args = parser.parse_args()
    train(train_args)
