import cv2
import pickle
from PIL import Image
import numpy as np

import torch
from torch import nn
from torchvision import transforms

from pose_detector.model.model import PoseDetector
from v_u_net.model.V_U_Net import CachedVUNet
from pose_drawer.pose_drawer import Pose_Drawer
import v_u_net.hyperparams as hp
from v_u_net.localise_joint_appearances import get_localised_joints


class Monkey:
    """
    Handles all the shapeshifting!
    https://jackiechanadventures.fandom.com/wiki/Monkey_Talisman
    """
    def __init__(self):
        self.use_cuda = True
        self.pose_drawer = Pose_Drawer()

        pose_model_base_path = 'pretrained_models/pose_detector.pt'
        pose_model = PoseDetector()
        pose_model.load_state_dict(torch.load(pose_model_base_path))
        pose_model = pose_model.eval()
        if self.use_cuda:
            pose_model = pose_model.cuda()
        self.pose_model = pose_model

        app_model_base_path = 'pretrained_models/v_u_net.pt'
        app_model = CachedVUNet()
        app_model.load_state_dict(torch.load(app_model_base_path))
        app_model = app_model.eval()
        if self.use_cuda:
            app_model = app_model.cuda()
        self.app_model = app_model
        # TODO: This is temporary and should be neatened up
        # Also think about getting the pose/keypoints directly from the input img
        app_img_path = 'test_imgs/test_appearance_img.jpg'
        joint_pos_path = 'test_imgs/test_appearance_joint_pos.pkl'
        app_img = Image.open(app_img_path)
        with open(joint_pos_path, 'rb') as in_f:
            app_joint_pos = pickle.load(in_f)
        self._generate_appearance_cache(app_img, app_joint_pos)

    @staticmethod
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
        canvas = np.zeros([256, 256, 3]).astype(int)
        canvas[:height, :width, :] = resized_img
        return canvas

    def transfer_appearance(self, pose_img):
        """
        This is the full shapeshift
        Args:
            pose_img (np array): The pose to transfer to
        """
        if pose_img is None:
            return

        pose_img = transforms.ToTensor()(pose_img).float()
        pose_img = pose_img / 256
        pose_img = pose_img.unsqueeze(0)
        if self.use_cuda:
            pose_img = pose_img.cuda()

        with torch.no_grad():
            gen_img = self.app_model(pose_img, self.app_vec_1, self.app_vec_2)

        gen_img = gen_img.squeeze(0).permute(1, 2, 0)
        gen_img = gen_img.detach().cpu().numpy()

        # WebRTC wants in range 0-256
        gen_img *= 256
        return gen_img

    def draw_pose_from_img(self, img):
        """
        Args:
            img (np array)
        Returns:
            pose_img (np array)
        """
        if img is None:
            return

        img = self.preprocess_img(img)
        img = img.astype('double')
        img_tensor = transforms.ToTensor()(img).float()

        if self.use_cuda:
            img_tensor = img_tensor.cuda()

        with torch.no_grad():
            _, pred_heat_maps = self.pose_model(img_tensor.view(1, 3, 256, 256))

        final_heat_maps = nn.functional.interpolate(pred_heat_maps[-1], scale_factor=4)
        final_heat_maps = final_heat_maps.view(18, 256, 256)
        final_heat_maps = final_heat_maps.cpu().detach().numpy()
        final_heat_maps = _zero_heat_map_edges(final_heat_maps)

        pose_img = self.pose_drawer.draw_pose_from_heatmaps(final_heat_maps)
        # WebRTC wants in range 0-256
        pose_img *= 256
        return pose_img

    def _generate_appearance_cache(self, app_img, app_joint_pos):
        """
        We don't need to recalculate the encoded appearance every
        single forward pass so just do it once.
        """
        app_encoder_inp = self._prep_app_encoder_inp(app_img, app_joint_pos)
        with torch.no_grad():
            cache = self.app_model.appearance_encoder(*app_encoder_inp)
        self.app_vec_1, self.app_vec_2, _, _ = cache

    def _prep_app_encoder_inp(self, app_img, app_img_joint_pos):
        """
        Prepares the data for input into the VUNet appearance encoder.
        Returns a tuple of:
            app_img (PyTorch tensor): Desired appearance img.
            app_img_pose (PyTorch tensor): Pose of the desired appearance
            localised_joints (PyTorch tensor): Zomed imgs of appearance
                                               img joints.
        """
        app_joint_pixels = (app_img_joint_pos*hp.image_edge_size).astype('int')
        localised_joints = get_localised_joints(app_img,
                                                hp.joints_to_localise,
                                                app_joint_pixels)

        app_img_pose = self.pose_drawer.draw_pose_from_keypoints(app_img_joint_pos)
        app_img = transforms.ToTensor()(app_img)
        app_img_pose = transforms.ToTensor()(app_img_pose).float()

        localised_joints = [transforms.ToTensor()(joint_img) for joint_img in localised_joints]
        localised_joints = torch.cat(localised_joints, dim=0).float()

        # Mock having batch size one to make dimensions work in model.
        app_img = app_img.unsqueeze(0)
        app_img_pose = app_img_pose.unsqueeze(0)
        localised_joints = localised_joints.unsqueeze(0)

        if self.use_cuda:
            app_img = app_img.cuda()
            app_img_pose = app_img_pose.cuda()
            localised_joints = localised_joints.cuda()

        return (app_img, app_img_pose, localised_joints)


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
    heat_map_tensor[:, :5, :] = np.zeros([18, 5, 256])
    heat_map_tensor[:, :, :5] = np.zeros([18, 256, 5])
    heat_map_tensor[:, -5:, :] = np.zeros([18, 5, 256])
    heat_map_tensor[:, :, -5:] = np.zeros([18, 256, 5])
    return heat_map_tensor
