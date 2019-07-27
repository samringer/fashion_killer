from pathlib import Path
import pickle
from random import sample, random

import cv2
import numpy as np
import torch
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

from utils import set_seeds


class AsosDataset(Dataset):
    """
    Dataset of the pairs of original and pose images for the
    clean asos data.
    """

    def __init__(self, root_data_dir='/home/sam/data/asos/1307_clean/train/',
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
        target_path = outfit_dir/str(pose_num)

        app_img = cv2.imread(app_path.as_posix())
        target_img = cv2.imread(target_path.as_posix())

        if random() < 0.5:
            # Randomly do an LR flip on app img
            app_img = cv2.flip(app_img, 1)
            app_pose_path = app_path.with_suffix('.revpose.jpg')
            app_kp_path = app_path.with_suffix('.kp')
        else:
            app_pose_path = app_path.with_suffix('.pose.jpg')
            app_kp_path = app_path.with_suffix('.revkp')

        if random() < 0.5:
            # Randomly do an LR flip on target img
            target_img = cv2.flip(target_img, 1)
            pose_path = target_path.with_suffix('.revpose.jpg')
            target_kp_path = target_path.with_suffix('.kp')
        else:
            pose_path = target_path.with_suffix('.pose.jpg')
            target_kp_path = target_path.with_suffix('.revkp')

        app_pose_img = cv2.imread(app_pose_path.as_posix())
        pose_img = cv2.imread(pose_path.as_posix())

        app_img = cv2.cvtColor(app_img, cv2.COLOR_BGR2RGB)
        app_pose_img = cv2.cvtColor(app_pose_img, cv2.COLOR_BGR2RGB)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        pose_img = cv2.cvtColor(pose_img, cv2.COLOR_BGR2RGB)

        app_img = preprocess_img(app_img)
        target_img = preprocess_img(target_img)

        app_img = ToTensor()(np.asarray(app_img) / 256)
        app_pose_img = ToTensor()(np.asarray(app_pose_img) / 256)
        target_img = ToTensor()(np.asarray(target_img) / 256)
        pose_img = ToTensor()(np.asarray(pose_img) / 256)

        # Prepare the keypoint heatmaps and cat to pose imgs
        with app_kp_path.open('rb') as in_f:
            app_kp = pickle.load(in_f)

        with target_kp_path.open('rb') as in_f:
            target_kp = pickle.load(in_f)

        app_hms = generate_kp_heatmaps(app_kp)
        target_hms = generate_kp_heatmaps(target_kp)
        app_pose_img = torch.cat([app_pose_img, app_hms])
        pose_img = torch.cat([pose_img, target_hms])

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
        canvas (np array): 256x256x3 image of preprocessed img.
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


def generate_kp_heatmaps(kps):
    """
    Generates the exponential decay heatmaps for all the keypoints.
    Args:
        kps (l o tuples): Coordinates of the keypoints.
    Returns:
        np_array (PyTorch tensor): Stacked heatmaps of all the keypoints
    """
    kp_tensor = torch.zeros([18, 256, 256])
    sigma = 5

    for i, keypoint in enumerate(kps):
        if keypoint == (0, 0):
            continue
        # TODO: Not sure where this error originated
        keypoint = (201 - keypoint[0], keypoint[1])

        x, y = np.arange(0, 256), np.arange(0, 256)
        xx, yy = np.meshgrid(x, y)
        xx, yy = xx - keypoint[0], yy - keypoint[1]
        z = np.exp(-(np.sqrt(np.square(xx) + np.square(yy))/np.square(sigma)))
        kp_tensor[i] = torch.from_numpy(z)
    return kp_tensor.double()
