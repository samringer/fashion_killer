import cv2
import numpy as np

import torch
from torch import nn
from torchvision import transforms

from pose_detector.model.model import PoseDetector
 #from v_u_net.model.v_u_net import CachedVUNet
#from asos_net.model.u_net import UNet
from pose_drawer.pose_drawer import PoseDrawer
from v_u_net.localise_joint_appearances import get_localised_joints
from GAN.model.u_net import UNet


class Monkey:
    """
    Handles all the shapeshifting!
    https://jackiechanadventures.fandom.com/wiki/Monkey_Talisman
    """

    use_cuda = torch.cuda.is_available()
    pose_drawer = PoseDrawer()

    pose_model_base_path = 'pretrained_models/pose_detector.pt'
    app_model_base_path = 'pretrained_models/11_05_fac_2_down_gan.pt'
    app_img_path = 'test_imgs/2204_asos.jpg'

    def __init__(self):
        pose_model = PoseDetector()
        if self.pose_model_base_path:
            pose_model.load_state_dict(torch.load(self.pose_model_base_path))
        self.pose_model = pose_model.eval()
        if self.use_cuda:
            self.pose_model = self.pose_model.cuda()

         #app_model = CachedVUNet()
        app_model = UNet()
        if self.app_model_base_path:
            app_model.load_state_dict(torch.load(self.app_model_base_path))
        self.app_model = app_model.eval()
        if self.use_cuda:
            self.app_model = self.app_model.cuda()

        # TODO: This is temporary and should be neatened up
        # app_img_path = 'test_imgs/test_appearance_img.jpg'
        app_img = cv2.imread(self.app_img_path)
        app_img = cv2.cvtColor(app_img, cv2.COLOR_BGR2RGB)
        app_img = self.preprocess_img(app_img)
        app_img = np.asarray(app_img) / 256
        # TODO: There is lots of replication in here thats in other methods
        # TODO: Work so this float is not needed
        app_tensor = transforms.ToTensor()(app_img).float()

        # Downsample as using smaller ims for now
        app_tensor = nn.MaxPool2d(kernel_size=2)(app_tensor)
        self.app_tensor = app_tensor.view(1, 3, 128, 128)

        if self.use_cuda:
            self.app_tensor = self.app_tensor.cuda()
         #self._generate_appearance_cache(app_img)

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
        canvas = np.zeros([256, 256, 3])
        canvas[:height, :width, :] = resized_img
        return canvas

    def transfer_appearance(self, pose_img):
        """
        This is the full shapeshift.
        Args:
            pose_img (np array): The pose to transfer to
        """
        if pose_img is None:
            return

        pose_tensor = transforms.ToTensor()(pose_img).float()
        pose_tensor = pose_tensor.unsqueeze(0)

        # Downsample as currently using smaller imgs
        pose_tensor = nn.MaxPool2d(kernel_size=2)(pose_tensor)
        if self.use_cuda:
            pose_tensor = pose_tensor.cuda()

        with torch.no_grad():
            gen_img = self.app_model(self.app_tensor, pose_tensor)

        gen_img = gen_img.squeeze(0).permute(1, 2, 0)
        gen_img = gen_img.detach().cpu().numpy()
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
        img_tensor = transforms.ToTensor()(img).float()

        if self.use_cuda:
            img_tensor = img_tensor.cuda()

        with torch.no_grad():
            pafs, heat_maps = self.pose_model(img_tensor.view(1, 3, 256, 256))

        heat_maps = nn.functional.interpolate(heat_maps[-1], scale_factor=4)
        heat_maps = heat_maps.view(18, 256, 256)
        heat_maps = heat_maps.cpu().detach().numpy()
        # TODO: Neaten up
        heat_maps[14] *= 0.8
        heat_maps[15] *= 0.8
        heat_maps[16] *= 0.8
        heat_maps[17] *= 0.8
        # TODO: potentilly remove
        # heat_maps = _zero_heat_map_edges(heat_maps)

        return self.pose_drawer.draw_pose_from_heat_maps(heat_maps)

    def _generate_appearance_cache(self, app_img):
        """
        We don't need to recalculate the encoded appearance every
        single forward pass so just do it once.
        """
        app_img = self.preprocess_img(app_img)
        # TODO: There is lots of replication in here thats in other methods
        app_tensor = transforms.ToTensor()(app_img).float()
        app_tensor = app_tensor.view(1, 3, 256, 256)

        if self.use_cuda:
            app_tensor = app_tensor.cuda()

        with torch.no_grad():
            _, heat_maps = self.pose_model(app_tensor)

        heat_maps = nn.functional.interpolate(heat_maps[-1], scale_factor=4)
        heat_maps = heat_maps.view(18, 256, 256)
        heat_maps = heat_maps.cpu().detach().numpy()
        # TODO:
        # heat_maps = _zero_heat_map_edges(heat_maps)

        joint_pos = self.pose_drawer.extract_keypoints_from_heat_maps(heat_maps)
        app_enc_inp = self._prep_app_encoder_inp(app_img, joint_pos)
        with torch.no_grad():
            cache = self.app_model.appearance_encoder(*app_enc_inp)
        self.app_vec_1, self.app_vec_2, _, _ = cache

    def _prep_app_encoder_inp(self, app_img, app_joint_pos):
        """
        Prepares the data for input into the VUNet appearance encoder.
        Returns a tuple of:
            app_img (PyTorch tensor): Desired appearance img.
            app_img_pose (PyTorch tensor): Pose of the desired appearance
            localised_joints (PyTorch tensor): Zomed imgs of appearance
                                               img joints.
        """
        localised_joints = get_localised_joints(app_img, app_joint_pos)

        app_img_pose = self.pose_drawer.draw_pose_from_keypoints(app_joint_pos)
        app_img = transforms.ToTensor()(app_img).float()
        app_img_pose = transforms.ToTensor()(app_img_pose).float()

        localised_joints = [transforms.ToTensor()(i).float()
                            for i in localised_joints]
        localised_joints = torch.cat(localised_joints).float()

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
