from os.path import join
import pickle

from absl import flags, app, logging
import torch
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, DistributedSampler
import pytorch_lightning as pl
from test_tube import Experiment

from app_transfer.model.generator import Generator
from app_transfer.dataset import AsosDataset
from app_transfer.model.perceptual_loss_vgg import PerceptualLossVGG
from utils import (prepare_experiment_dirs,
                   get_tb_logger,
                   set_seeds)

FLAGS = flags.FLAGS


class MyLightning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.generator = Generator()
        self.perceptual_loss_vgg = PerceptualLossVGG()
        #self.perceptual_loss_vgg.eval()

    def forward(self, app_img, app_pose_img, pose_img):
        return self.generator(app_img, app_pose_img, pose_img)

    def training_step(self, batch, batch_nb):
        app_img = batch['app_img']
        app_pose_img = batch['app_pose_img']
        target_img = batch['target_img']
        pose_img = batch['pose_img']

        gen_img = self.forward(app_img, app_pose_img, pose_img)

        self.experiment.add_image('generated_img', gen_img[0], batch_nb)

        l1_loss = nn.L1Loss()(gen_img, target_img)
        perceptual_loss = 0.02 * self.perceptual_loss_vgg(gen_img,
                                                          target_img)
        loss = l1_loss + perceptual_loss
        return {'loss': loss}

    def configure_optimizers(self):
        return optim.Adam(self.generator.parameters(),
                          lr=FLAGS.learning_rate, betas=(0.5, 0.999))

    @pl.data_loader
    def tng_dataloader(self):
        dataset = AsosDataset(root_data_dir=FLAGS.data_dir,
                              overtrain=FLAGS.overtrain)
        dataloader = DataLoader(dataset, batch_size=FLAGS.batch_size,
                                shuffle=True,
                                num_workers=5, pin_memory=True)
        return dataloader


def train(unused_argv):
    set_seeds()
    model = MyLightning()
    exp_dir = join(FLAGS.task_path, FLAGS.exp_name)
    exp = Experiment(save_dir=exp_dir, create_git_tag=True)
    exp.tag({
        'batch_size': FLAGS.batch_size,
        'learning_rate': FLAGS.learning_rate,
        'description': 'full size img training'
    })

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=join(exp_dir, 'weights'),
        save_best_only=False,
        verbose=1,
        period=1
    )

    trainer = pl.Trainer(max_nb_epochs=3,
                         distributed_backend='dp',
                         gpus=[0, 1],
                         experiment=exp,
                         checkpoint_callback=checkpoint_callback)
    trainer.fit(model)


if __name__ == "__main__":
    app.run(train)
