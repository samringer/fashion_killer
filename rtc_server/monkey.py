import cv2
import numpy as np

import torch
from torch import nn
from torchvision import transforms, models

from pose_detector.model.model import PoseDetector
from pose_drawer.pose_drawer import PoseDrawer
from v_u_net.localise_joint_appearances import get_localised_joints
from DeformGAN.model.generator import Generator

class Monkey:
    """
    Handles all the shapeshifting!
    https://jackiechanadventures.fandom.com/wiki/Monkey_Talisman
    """

    use_cuda = torch.cuda.is_available()
    pose_drawer = PoseDrawer()

    pose_model_base_path = 'pretrained_models/pose_detector.pt'
    app_model_base_path = 'pretrained_models/2306_fac_2_down.pt'
    rcnn_base_path = 'pretrained_models/keypointrcnn_resnet50_fpn_coco-9f466800.pth'
    app_img_path = 'test_imgs/2406_input_app.jpg'
    app_pose_img_path = 'test_imgs/2406_input_app.pose.jpg'

    def __init__(self):
        pose_model = models.detection.keypointrcnn_resnet50_fpn()
        pose_model.load_state_dict(torch.load(self.rcnn_base_path))
        self.pose_model = pose_model.eval()
        if self.use_cuda:
            self.pose_model = self.pose_model.cuda()

        #app_model = Generator()
        #if self.app_model_base_path:
        #    app_model.load_state_dict(torch.load(self.app_model_base_path))
        #self.app_model = app_model.eval()
        #if self.use_cuda:
        #    self.app_model = self.app_model.cuda()

        # TODO: This is temporary and should be neatened up
        #app_img = cv2.imread(self.app_img_path)
        #app_img = cv2.cvtColor(app_img, cv2.COLOR_BGR2RGB)
        #app_img = self.preprocess_img(app_img)
        #app_img = np.asarray(app_img) / 256
        # TODO: There is lots of replication in here thats in other methods
        #app_tensor = transforms.ToTensor()(app_img).float()

        # Downsample as using smaller ims for now
        #app_tensor = nn.MaxPool2d(kernel_size=2)(app_tensor)
        #self.app_tensor = app_tensor.view(1, 3, 128, 128)

        # TODO: This is temporary and should be neatened up
        #app_pose_img = cv2.imread(self.app_pose_img_path)
        #app_pose_img = cv2.cvtColor(app_pose_img, cv2.COLOR_BGR2RGB)
        #app_pose_img = self.preprocess_img(app_pose_img)
        #app_pose_img = np.asarray(app_pose_img) / 256
        # TODO: There is lots of replication in here thats in other methods
        #app_pose_tensor = transforms.ToTensor()(app_pose_img).float()

        # Downsample as using smaller ims for now
        #app_pose_tensor = nn.MaxPool2d(kernel_size=2)(app_pose_tensor)
        #self.app_pose_tensor = app_pose_tensor.view(1, 3, 128, 128)

        #if self.use_cuda:
        #    self.app_tensor = self.app_tensor.cuda()
        #    self.app_pose_tensor = self.app_pose_tensor.cuda()
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
        # TODO: This shouldn't be constant being put and pulled off GPU
        pose_tensor = nn.MaxPool2d(kernel_size=2)(pose_tensor)
        if self.use_cuda:
            pose_tensor = pose_tensor.cuda()

        with torch.no_grad():
            gen_img = self.app_model(self.app_tensor,
                                     self.app_pose_tensor, pose_tensor)

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
        img_tensor = img_tensor.view(1, 3, 256, 256)
        img_tensor = nn.functional.interpolate(img_tensor,
                                               size=(800, 800))

        if self.use_cuda:
            img_tensor = img_tensor.cuda()

        with torch.no_grad():
            model_out = self.pose_model(img_tensor)

        keypoints = extract_keypoints(model_out)
        return self.pose_drawer.draw_pose_from_keypoints(keypoints), keypoints

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


def extract_keypoints(pose_model_out):
    """
    Used for preparing the keypoints the are the output of
    the pretrained torchvision model.
    """
    keypoints = pose_model_out[0]['keypoints'].cpu().numpy()
    if len(keypoints) == 0:
        return [(0, 0) for _ in range(18)]
    keypoints = keypoints[0]
    keypoints_score = pose_model_out[0]['keypoints_scores'].cpu().numpy()[0]
    filtered_keypoints = [tuple([256*j/800 for j in kp[:2]])
                          if keypoints_score[i] > 1.5 else (0, 0)
                          for i, kp in enumerate(keypoints)]
    filtered_keypoints = add_neck_keypoint(filtered_keypoints)
    return filtered_keypoints


def add_neck_keypoint(keypoints):
    """
    COCO by default does not contain a neck keypoint.
    This function adds it in as an average of the left
    and right shoulders if both are found, (0, 0) otherwise.
    Args:
        keypoints (l o tuples)
    Returns:
        keypoints (l o tuples): Keypoints with neck added in.
    """
    r_shoulder_keypoint = keypoints[5]
    l_shoulder_keypoint = keypoints[6]
    if r_shoulder_keypoint != (0, 0) and l_shoulder_keypoint != (0, 0):
        neck_keypoint_x = (r_shoulder_keypoint[0] + l_shoulder_keypoint[0])//2
        neck_keypoint_y = (r_shoulder_keypoint[1] + l_shoulder_keypoint[1])//2
        neck_keypoint = (neck_keypoint_x, neck_keypoint_y)
    else:
        neck_keypoint = (0, 0)
    keypoints.insert(1, neck_keypoint)
    return keypoints
