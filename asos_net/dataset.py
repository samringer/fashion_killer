from pathlib import Path
from random import sample

import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset

from utils import set_seeds


class AsosDataset(Dataset):
    """
    Dataset of the pairs of original and pose images for the
    clean asos data.
    """

    def __init__(self, root_data_dir='/home/sam/data/asos/train_clean/',
                 overtrain=False):
        """
        Args:
            root_data_dir (str): Path to directory containing data.
            overtrain (bool): If True, same img is always returned.
        """
        self.overtrain = overtrain
        root_data_dir = Path(root_data_dir)

        self.outfit_dirs = list(root_data_dir.glob('*'))

    def __len__(self):
        # Note the concept of 'length' for this dataset is
        # a bit fuzzy.
        return len(self.outfit_dirs)

    def __getitem__(self, index):
        if self.overtrain:
            index = 0
            set_seeds()

        outfit_dir = self.outfit_dirs[index]

        # Sample such that we will not use an image with it's
        # corresponding pose.
        orig_num, pose_num = sample(list(range(3)), 2)

        orig_path = outfit_dir / str(orig_num)
        pose_path = (outfit_dir / str(pose_num)).with_suffix('.pose.jpg')
        target_path = outfit_dir / str(pose_num)

        orig_img = cv2.imread(orig_path.as_posix())
        pose_img = cv2.imread(pose_path.as_posix())
        target_img = cv2.imread(target_path.as_posix())

        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        pose_img = cv2.cvtColor(pose_img, cv2.COLOR_BGR2RGB)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

        # Note pose_img has already been preprocessed during cleaning.
        orig_img = preprocess_img(orig_img)
        target_img = preprocess_img(target_img)

        orig_img = transforms.ToTensor()(np.asarray(orig_img) / 256)
        pose_img = transforms.ToTensor()(np.asarray(pose_img) / 256)
        target_img = transforms.ToTensor()(np.asarray(target_img) / 256)

        return {'orig_img': orig_img,
                'pose_img': pose_img,
                'target_img': target_img}


def preprocess_img(img):
    """
    Resized an image and pads it so final size
    is 256x256.
    Args:
        img (np array)
    Returns:
        canvas (np array): 256x256 image of preprocessed img.
    """
    if img is None:
        return

    img_width, img_height, _ = img.shape
    current_max_dim = max(img_width, img_height)
    scale_factor = 256 / current_max_dim
    resized_img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
    height, width, _ = resized_img.shape
    canvas = np.zeros([256, 256, 3])
    canvas[:height, :width, :] = resized_img
    return canvas
