import unittest, pickle
from os.path import join, realpath, dirname

from PIL import Image
import numpy as np
import torch

from rtc_server.monkey import Monkey

UNITTESTS_DIRECTORY = dirname(realpath(__file__))


class TestMonkey(unittest.TestCase):

    def setUp(self):
        self.datadir = join(UNITTESTS_DIRECTORY, 'data/monkey')
        in_img_path = join(self.datadir, 'test_img.jpg')
        in_img = Image.open(in_img_path)
        self.in_img = np.array(in_img)

    def test_preprocess_img(self):
        """
        Test ability to perform the standard preprocessing on
        an input image.
        """
        monkey = Monkey()
        preprocessed_img = monkey.preprocess_img(self.in_img)

        ground_truth_path = join(self.datadir, 'preprocessed_img.pkl')
        with open(ground_truth_path, 'rb') as in_f:
            ground_truth = pickle.load(in_f)

        self.assertEqual(preprocessed_img.shape, (256, 256, 3))
        self.assertEqual(preprocessed_img.tolist(), ground_truth.tolist())

    def test_draw_pose_from_img(self):
        """
        Test ability of monkey to draw a pose img from an
        input img.
        """
        monkey = Monkey(transform='pose')
        pose_img = monkey.draw_pose_from_img(self.in_img)
        self.assertEqual(pose_img.shape, (256, 256, 3))

    def test_transfer_appearance(self):
        """
        Test ability of monkey to do end to end appearance transfer.
        Requires drawing a pose img as an intermediate step.
        """
        monkey = Monkey(transform='app_transfer')
        pose_img = monkey.draw_pose_from_img(self.in_img)
        appearance_img = monkey.transfer_appearance(pose_img)
        self.assertEqual(appearance_img.shape, (256, 256, 3))

    def test_prep_app_encoder_inp(self):
        """
        As part of preparing the cache when doing appearance transfer,
        monkey prepares appearance data to be fed into the VUNet
        appearance encoder. Test this is done correctly.
        """
        monkey = Monkey(transform='app_transfer')

        joint_pos_path = join(self.datadir, 'test_joint_pos.pkl')
        with open(joint_pos_path, 'rb') as in_f:
            app_joint_pos = pickle.load(in_f)

        output = monkey._prep_app_encoder_inp(self.in_img, app_joint_pos)

        app_img, app_img_pose, localised_joints = output

        # Check the shapes are correct.
        self.assertEqual(app_img.shape,
                         torch.Size([1, 3, 256, 256]))
        self.assertEqual(app_img_pose.shape,
                         torch.Size([1, 3, 256, 256]))
        self.assertEqual(localised_joints.shape,
                         torch.Size([1, 21, 256, 256]))


if __name__ == "__main__":
    unittest.main()
