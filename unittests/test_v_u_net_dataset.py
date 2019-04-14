import unittest
from os.path import join, dirname, realpath
import numpy as np

import cv2
from PIL import Image

from v_u_net.dataset import VUNetDataset, _rearrange_keypoints


UNITTESTS_DIR = dirname(realpath(__file__))


class VUNetDatasetTester(unittest.TestCase):
    """
    Unittest the dataset used for the VUNet.
    """

    def setUp(self):
        self.datadir = join(UNITTESTS_DIR, 'data/v_u_net')
        self.dataset = VUNetDataset(self.datadir)

    def test_dataset_length(self):
        self.assertEqual(len(self.dataset), 1)

    def test_getitem(self):
        datapoint = next(iter(self.dataset))

        app_img_shape = list(datapoint['app_img'].shape)
        self.assertEqual(app_img_shape, [3, 256, 256])

        pose_img_shape = list(datapoint['pose_img'].shape)
        self.assertEqual(pose_img_shape, [3, 256, 256])

        localised_joints_shape = list(datapoint['localised_joints'].shape)
        self.assertEqual(localised_joints_shape, [21, 256, 256])


class ImageProcessingTester(unittest.TestCase):
    """
    Tests general image processing of input data.
    """
    def setUp(self):
        self.datadir = join(UNITTESTS_DIR, 'data/v_u_net')
        self.dataset = VUNetDataset(root_data_dir=self.datadir)

    def test_generate_pose_img(self):
        from pose_drawer.pose_drawer import Pose_Drawer

        joint_raw_pos = self.dataset.data['joints'][0]
        joint_raw_pos = _rearrange_keypoints(joint_raw_pos)
        joint_pixel_pos = (joint_raw_pos*256).astype('int')
        pose_img = Pose_Drawer().draw_pose_from_keypoints(joint_pixel_pos)

        # Need a small adjustment to account for cv2
        # saving things a bit wierdly.
        pose_img = (pose_img*256).astype('int')

        ground_truth_path = join(self.datadir, 'test_pose.png')
        ground_truth = cv2.imread(ground_truth_path)

        self.assertTrue((pose_img == ground_truth).all())

    def test_get_localised_joints(self):
        """
        To help the model train, a copy of a crop of some
        joints is fed into the model. I have called this
        process 'localising'.
        """
        from v_u_net.localise_joint_appearances import get_localised_joints
        img_path = join(self.datadir, 'test_input.jpg')
        img = Image.open(img_path)
        img = np.asarray(img)

        joints_to_localise = self.dataset.joints_to_localise
        joint_raw_pos = self.dataset.data['joints'][0]
        joint_raw_pos = _rearrange_keypoints(joint_raw_pos)
        joint_pixel_pos = (joint_raw_pos*256).astype('int')

        localised_joints = get_localised_joints(img,
                                                joints_to_localise,
                                                joint_pixel_pos)

        self.assertEqual(len(localised_joints), len(joints_to_localise))
        for localised_joint in localised_joints:
            self.assertEqual(localised_joint.shape, (256, 256, 3))


if __name__ == "__main__":
    unittest.main()
