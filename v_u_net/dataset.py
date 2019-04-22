from pathlib import Path
import pickle

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset

from v_u_net.localise_joint_appearances import get_localised_joints
from pose_drawer.pose_drawer import PoseDrawer

image_edge_size = 256

# Which joints we want to localise and feed into appearance encoder
from pose_drawer.pose_settings import JointType
joints_to_localise = [JointType.Nose,
                      JointType.RightShoulder,
                      JointType.LeftShoulder,
                      JointType.RightElbow,
                      JointType.LeftElbow,
                      JointType.RightHand,
                      JointType.LeftHand]


class VUNetDataset(Dataset):
    """
    Dataset consists of pairs of original and pose extracted images
    """

    def __init__(self, root_data_dir='/home/sam/data/deepfashion',
                 overtrain=False):
        """
        Args:
            root_data_dir (str): Path to directory containing data.
            overtrain (bool): If True, same img is always returned.
        """
        self.overtrain = overtrain

        self.root_data_dir = Path(root_data_dir)
        index_path = self.root_data_dir/'index.p'
        with open(str(index_path), 'rb') as in_f:
            self.data = pickle.load(in_f)

        self.joints_to_localise = [joint.value for joint in joints_to_localise]
        self.pose_drawer = PoseDrawer()

    def __len__(self):
        return len(self.data['imgs'])

    def __getitem__(self, index):
        if self.overtrain:
            index = 0

        orig_img, pose_img, localised_joints = self._prepare_input_data(index)
        orig_img = transforms.ToTensor()(orig_img)
        pose_img = transforms.ToTensor()(pose_img)
        localised_joints = [transforms.ToTensor()(i)
                            for i in localised_joints]
        localised_joints = torch.cat(localised_joints)

        return {'app_img': orig_img,
                'pose_img': pose_img,
                'localised_joints': localised_joints}

    def _prepare_input_data(self, index):
        """
        Prepares data for input into the model.
        DOES NOT transform into PyTorch tensor and normalise.
        Args:
            index (int): index of datapoint to prepare.
        """
        img_path = self.root_data_dir/self.data['imgs'][index]
        orig_img = Image.open(str(img_path))
        orig_img = np.asarray(orig_img) / 256

        joint_raw_pos = self.data['joints'][index]
        joint_raw_pos = _rearrange_keypoints(joint_raw_pos)
        joint_pixel_pos = (joint_raw_pos*image_edge_size).astype('int')

        pose_img = self.pose_drawer.draw_pose_from_keypoints(joint_pixel_pos)
        localised_joints = get_localised_joints(orig_img,
                                                self.joints_to_localise,
                                                joint_pixel_pos)

        return orig_img, pose_img, localised_joints


def _rearrange_keypoints(keypoints):
    """
    The order of the joints in the keypoints used by the
    deepfashion dataset is different from that used for COCO.
    Rearrange deepfashion keypoints to be in order of COCO.
    Args:
        keypoints (list): list of the keypoints in x y list pairs
    """
    new_keypoints = np.zeros_like(keypoints)
    for old_pos, new_pos in DEEPFASHION_COCO_MAPPING.items():
        new_keypoints[new_pos] = keypoints[old_pos]
    return new_keypoints


# Comments are the originals
DEEPFASHION_COCO_MAPPING = {
    0: 0,  # nose
    1: 1,  # neck
    2: 7,  # right shoulder
    3: 9,  # right elbow
    4: 11,  # right hand
    5: 6,  # left shoulder
    6: 8,  # left elbow
    7: 10,  # left hand
    8: 13,  # right waist
    9: 15,  # right knee
    10: 17 , # right foot
    11: 12 , # left waist
    12: 14 , # left knee
    13: 16 , # left foot
    14: 3,  # right eye
    15: 2,  # left eye
    16: 3,  # right ear
    17: 4,  # left ear
}
