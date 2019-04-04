import cv2
import numpy as np

import torch
from torch import nn
from torchvision import transforms

from pose_detector.model.model import Model
from pose_drawer.pose_drawer import Pose_Drawer


class Monkey:
    """
    Handles all the shapeshifting!
    https://jackiechanadventures.fandom.com/wiki/Monkey_Talisman
    """
    def __init__(self):
        self.use_cuda = True

        self.pose_drawer = Pose_Drawer()

        model_base_path = 'pretrained_models/pose_detector.pt'
        model = Model()
        model.load_state_dict(torch.load(model_base_path))
        model = model.eval()
        if self.use_cuda:
            model = model.cuda()
        self.model = model

    def draw_pose_from_img(self, img):
        """
        Args:
            img (numpy array)
        """
        if img is None:
            return

        img = _preprocess_input_img(img)
        img = img.astype('double')
        img_tensor = transforms.ToTensor()(img).float()

        if self.use_cuda:
            img_tensor = img_tensor.cuda()

        with torch.no_grad():
            _, pred_heat_maps = self.model(img_tensor.view(1, 3, 256, 256))

        final_heat_maps = nn.functional.interpolate(pred_heat_maps[-1], scale_factor=4)
        final_heat_maps = final_heat_maps.view(18, 256, 256)
        final_heat_maps = final_heat_maps.cpu().detach().numpy()
        final_heat_maps = _zero_heat_map_edges(final_heat_maps)

        # TODO: think don't need as much overkill in this line.
        # Think don't need list comp as long as its a np array.
        heat_map_list = [final_heat_maps[i] for i in range(final_heat_maps.shape[0])]
        pose_img = self.pose_drawer.draw_pose_from_heatmaps(heat_map_list)
        return pose_img


def _zero_heat_map_edges(heat_map_tensor):
    """
    There is an error where heat is very high around edge
    of some heat maps. This is a hack to fix that and should
    ideally not be permenant.
    Args:
        heat_map_tensor (PyTorch tensor): Of size (18, 256, 256)
    Returns:
        heat_map_tensor (PyTorch tensor): Of size (18, 256, 256)
    """
    heat_map_tensor[:, :8, :] = np.zeros([18, 8, 256])
    heat_map_tensor[:, :, :8] = np.zeros([18, 256, 8])
    heat_map_tensor[:, -8:, :] = np.zeros([18, 8, 256])
    heat_map_tensor[:, :, -8:] = np.zeros([18, 256, 8])
    return heat_map_tensor


def _preprocess_input_img(img):
    """
    Resized an image and pads it so final size
    is 256x256.
    Args:
        img (np array)
    Returns:
        canvas (np array): 256x256 image of preprocessed img.
    """
    img_width, img_height, _ = img.shape
    current_max_dim = max(img_width, img_height)
    scale_factor = 256 / current_max_dim
    resized_img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
    height, width, _ = resized_img.shape
    canvas = np.zeros([256, 256, 3]).astype(int)
    canvas[:height, :width, :] = resized_img
    return canvas
