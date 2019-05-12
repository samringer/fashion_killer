from pathlib import Path
import pickle
from random import sample

import cv2
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

from utils import set_seeds


class AsosDataset(Dataset):
    """
    Dataset of the pairs of original and pose images for the
    clean asos data.
    """

    def __init__(self, root_data_dir='/home/sam/data/asos/1205_test/',
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
        app_hm_path = (app_path).with_suffix('.pose.hm')
        pose_path = outfit_dir/str(pose_num)
        pose_hm_path = (pose_path).with_suffix('.pose.hm')

        app_img = cv2.imread(app_path.as_posix())
        pose_img = cv2.imread(pose_path.as_posix())
        with open(app_hm_path.as_posix(), 'rb') as in_f:
            app_hms = pickle.load(in_f)
        with open(pose_hm_path.as_posix(), 'rb') as in_f:
            pose_hms = pickle.load(in_f)

        app_img = cv2.cvtColor(app_img, cv2.COLOR_BGR2RGB)
        pose_img = cv2.cvtColor(pose_img, cv2.COLOR_BGR2RGB)

        app_img = preprocess_img(app_img)
        pose_img = preprocess_img(pose_img)

        app_img = transforms.ToTensor()(np.asarray(app_img) / 256)
        pose_img = transforms.ToTensor()(np.asarray(pose_img) / 256)

        return {'app_img': app_img.float(),
                'pose_img': pose_img.float(),
                'app_hms': app_hms,
                'pose_hms': pose_hms}


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
