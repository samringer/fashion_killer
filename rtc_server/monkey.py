import cv2
import pickle
from pathlib import Path
import numpy as np

import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.models.detection import keypointrcnn_resnet50_fpn

from pose_drawer.pose_drawer import PoseDrawer
from app_transfer.model.cached_generator import CachedGenerator
from app_transfer.dataset import preprocess_img

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Monkey:
    """
    Handles all the shapeshifting!
    https://jackiechanadventures.fandom.com/wiki/Monkey_Talisman
    """
    generator_base_path = 'pretrained_models/27_07_placeholder.pt'
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

    def draw_pose_from_img(self, img):
        """
        Args:
            img (np array)
        Returns:
            pose_img (np array)
        """
        if img is None:
            return None, None

        img = transforms.ToTensor()(img).view(1, 3, 256, 256).float()
        img = nn.functional.interpolate(img, size=(800, 800))
        img = img.to(device)

        with torch.no_grad():
            model_output = self.pose_model(img)

        keypoints = extract_keypoints(model_output)
        pose = self.pose_drawer.draw_pose_from_keypoints(keypoints)
        return pose, keypoints

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


def extract_keypoints(pose_model_out):
    """
    Used for preparing the keypoints the are the output of
    the pretrained torchvision model.
    """
    keypoints = pose_model_out[0]['keypoints'].cpu().numpy()
    if not keypoints.size:
        return [(0, 0) for _ in range(18)]
    keypoints = keypoints[0]
    keypoints_score = pose_model_out[0]['keypoints_scores'].cpu().numpy()[0]
    # 1.5 is arbritrarily chosen threshold
    # Pose detector outputs in 800x800 coordinated but we want 256x256
    filtered_keypoints = [tuple([256*j/800 for j in kp[:2]])
                          if keypoints_score[i] > 1.5 else (0, 0)
                          for i, kp in enumerate(keypoints)]
    filtered_keypoints = add_neck_keypoint(filtered_keypoints)
    return filtered_keypoints


def add_neck_keypoint(keypoints):
    """
    PyTorch pose detector does not return a neck keypoint.
    Add it in as an average of the left and right shoulders
    if both are found, (0, 0) otherwise.
    Args:
        keypoints (l o tuples)
    Returns:
        keypoints (l o tuples): Keypoints with neck added in.
    """
    r_shoulder_kp = keypoints[5]
    l_shoulder_kp = keypoints[6]
    if r_shoulder_kp != (0, 0) and l_shoulder_kp != (0, 0):
        neck_kp_x = (r_shoulder_kp[0] + l_shoulder_kp[0])//2
        neck_kp_y = (r_shoulder_kp[1] + l_shoulder_kp[1])//2
        neck_kp = (neck_kp_x, neck_kp_y)
    else:
        neck_kp = (0, 0)
    keypoints.insert(1, neck_kp)
    return keypoints


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

        x, y = np.arange(0, 256), np.arange(0, 256)
        xx, yy = np.meshgrid(x, y)
        xx, yy = xx - keypoint[0], yy - keypoint[1]
        z = np.exp(-(np.sqrt(np.square(xx) + np.square(yy))/np.square(sigma)))
        kp_tensor[i] = torch.from_numpy(z)
    return kp_tensor.float()
