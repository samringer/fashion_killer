import cv2
import pickle
from pathlib import Path
from copy import deepcopy
import numpy as np

import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.models.detection import keypointrcnn_resnet50_fpn

from pose_drawer.pose_drawer import PoseDrawer
from app_transfer.model.cached_generator import CachedGenerator
from app_transfer.dataset import preprocess_img
from pose_drawer.keypoints import KeyPoints

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Monkey:
    """
    Handles all the shapeshifting!
    https://jackiechanadventures.fandom.com/wiki/Monkey_Talisman
    """
    generator_base_path = 'pretrained_models/01_08_model.pt'
    rcnn_base_path = 'pretrained_models/pytorch_pose_detector.pt'

    def __init__(self):
        self.pose_drawer = PoseDrawer()

        pose_model = keypointrcnn_resnet50_fpn(pretrained_backbone=False)
        pose_model.load_state_dict(torch.load(self.rcnn_base_path))
        self.pose_model = pose_model.eval().to(device)

        generator = CachedGenerator()
        generator.load_state_dict(torch.load(self.generator_base_path))
        self.generator = generator.to(device)
        self.load_appearance_img(Path('rtc_server/assets/0'))

    def load_appearance_img(self, app_img_root):
        """
        Load an appearance image into the generator.
        Assumes that there is ".jpg", ".pose.jpg" and
        ".kp" files for the given image root
        Args:
            app_img_root (Pathlib object)
        """
        app_img_path = app_img_root.with_suffix('.jpg')
        app_pose_img_path = app_img_root.with_suffix('.pose.jpg')
        app_kp_path = app_img_root.with_suffix('.kp')

        app_img = cv2.imread(app_img_path.as_posix())
        app_pose_img = cv2.imread(app_pose_img_path.as_posix())

        app_img = preprocess_img(app_img)
        app_pose_img = preprocess_img(app_pose_img)

        # Prepare the keypoint heatmaps and cat to pose imgs
        with app_kp_path.open('rb') as in_f:
            app_kp = pickle.load(in_f)
        app_heatmaps = generate_kp_heatmaps(app_kp)
        app_pose_img = torch.cat([app_pose_img, app_heatmaps])

        app_img = app_img.view(1, 3, 256, 256).to(device)
        app_pose_img = app_pose_img.view(1, 21, 256, 256).to(device)
        self.generator.load_cache(app_img, app_pose_img)

    def draw_pose_from_img(self, img, kps=None):
        """
        Args:
            img (np array)
            kps (KeyPoints object): The kps from the previous frame
        Returns:
            pose_img (np array)
        """
        if img is None:
            return None, None

        if kps is None:
            # Initial condition where we have no keypoints
            kps = KeyPoints()

        img = transforms.ToTensor()(img).view(1, 3, 256, 256).float()
        img = nn.functional.interpolate(img, size=(800, 800))
        img = img.to(device)

        with torch.no_grad():
            model_output = self.pose_model(img)

        kps.update_markov_model(model_output)

        # TODO: Work out why this is needed
        pose = self.pose_drawer.draw_pose_from_keypoints(kps)
        return pose, kps

    def transfer_appearance(self, pose_img, keypoints):
        """
        This is the full shapeshift.
        Args:
            pose_img (np array): The pose to transfer to
            keypoints (l o tuples): Keypoints of this desired pose
        """
        if pose_img is None or keypoints is None:
            return

        pose_img = transforms.ToTensor()(pose_img).float()
        heatmaps = generate_kp_heatmaps(keypoints).float()
        pose_img = torch.cat([pose_img, heatmaps])
        pose_img = pose_img.unsqueeze(0).to(device)

        with torch.no_grad():
            gen_img = self.generator(pose_img)

        gen_img = gen_img.squeeze(0).permute(1, 2, 0)
        gen_img = gen_img.detach().cpu().numpy()
        return gen_img


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
        # TODO: Need to track this bug down as means I have to hack
        # monkey to compensate
        #keypoint = (201 - keypoint[0], keypoint[1])

        x, y = np.arange(0, 256), np.arange(0, 256)
        xx, yy = np.meshgrid(x, y)
        xx, yy = xx - keypoint[0], yy - keypoint[1]
        z = np.exp(-(np.sqrt(np.square(xx) + np.square(yy))/np.square(sigma)))
        kp_tensor[i] = torch.from_numpy(z)
    return kp_tensor.float()
