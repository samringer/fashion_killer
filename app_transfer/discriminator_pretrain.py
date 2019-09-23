from os.path import join

from absl import flags, app, logging
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from app_transfer.model.generator import Generator
from app_transfer.model.discriminator import Discriminator
from app_transfer.dataset import AsosDataset
from utils import (prepare_experiment_dirs,
                   set_seeds,
                   device)


FLAGS = flags.FLAGS
flags.DEFINE_string("gen_path", None, "Which generator to train discriminator with")


def train(unused_argv):
    """
    Pretrains a discriminator using a pretrained generator
    before joint training.
    """
    models_path = prepare_experiment_dirs()
    set_seeds(128)

    generator = Generator()
    generator.load_state_dict(torch.load(FLAGS.gen_path))
    generator = generator.to(device)
    discriminator = Discriminator().to(device)

    dataset = AsosDataset(root_data_dir=FLAGS.data_dir,
                          overtrain=FLAGS.over_train)
    dataloader = DataLoader(dataset, batch_size=FLAGS.batch_size,
                            shuffle=True, num_workers=6, pin_memory=True)

    d_optimizer = optim.Adam(discriminator.parameters(),
                             lr=FLAGS.learning_rate, betas=(0.5, 0.999))

    for step_num, batch in enumerate(dataloader):
        if step_num >= 1000:
            save_path = join(models_path, 'final.pt')
            torch.save(discriminator.state_dict(), save_path)
            return

        app_img = batch['app_img'].to(device)
        app_pose_img = batch['app_pose_img'].to(device)
        target_img = batch['target_img'].to(device)
        pose_img = batch['pose_img'].to(device)

        app_img = nn.MaxPool2d(kernel_size=2)(app_img)
        app_pose_img = nn.MaxPool2d(kernel_size=2)(app_pose_img)
        target_img = nn.MaxPool2d(kernel_size=2)(target_img)
        pose_img = nn.MaxPool2d(kernel_size=2)(pose_img)

        with torch.no_grad():
            gen_img = generator(app_img, app_pose_img, pose_img)

        real_score = discriminator(app_img, app_pose_img, pose_img,
                                   target_img)
        gen_score = discriminator(app_img, app_pose_img, pose_img,
                                  gen_img)
        real_loss = (nn.ReLU()(1. - real_score)).mean()
        gen_loss = (nn.ReLU()(1. + gen_score)).mean()
        loss = real_loss + gen_loss

        d_optimizer.zero_grad()
        loss.backward()
        d_optimizer.step()

        logging.info(f"{step_num} {loss.item():.4f}")


if __name__ == "__main__":
    app.run(train)
