from pathlib import Path
import pickle
from random import sample

import cv2
import numpy as np
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

from utils import set_seeds


class AsosDataset(Dataset):
    """
    Dataset of the pairs of original and pose images for the
    clean asos data.
    """

    def __init__(self, root_data_dir='/home/sam/data/asos/2604_clean/train/',
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
            set_seeds()  # Needed so same app_img always picked.

        outfit_dir = self.outfit_dirs[index]

        # Sample such that we will not use an appearance image with it's
        # corresponding pose.
        app_num, pose_num = sample(list(range(4)), 2)

        app_path = outfit_dir/str(app_num)
        app_pose_path = (app_path).with_suffix('.pose.jpg')
        target_path = outfit_dir/str(pose_num)
        pose_path = (target_path).with_suffix('.pose.jpg')

        app_img = cv2.imread(app_path.as_posix())
        app_pose_img = cv2.imread(app_pose_path.as_posix())
        target_img = cv2.imread(target_path.as_posix())
        pose_img = cv2.imread(pose_path.as_posix())

        app_img = cv2.cvtColor(app_img, cv2.COLOR_BGR2RGB)
        app_pose_img = cv2.cvtColor(app_pose_img, cv2.COLOR_BGR2RGB)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        pose_img = cv2.cvtColor(pose_img, cv2.COLOR_BGR2RGB)

        app_img = preprocess_img(app_img)
        app_pose_img = preprocess_img(app_pose_img)
        target_img = preprocess_img(target_img)
        pose_img = preprocess_img(pose_img)

        app_img = ToTensor()(np.asarray(app_img) / 256)
        app_pose_img = ToTensor()(np.asarray(app_pose_img) / 256)
        target_img = ToTensor()(np.asarray(target_img) / 256)
        pose_img = ToTensor()(np.asarray(pose_img) / 256)

        return {'app_img': app_img.float(),
                'app_pose_img': app_pose_img.float(),
                'target_img': target_img.float(),
                'pose_img': pose_img.float()}


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
