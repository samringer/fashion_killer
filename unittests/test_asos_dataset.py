import unittest
from os.path import join, dirname, realpath
import numpy as np

import cv2
from PIL import Image

from asos_net.dataset import AsosDataset

UNITTESTS_DIR = dirname(realpath(__file__))


class AsosDatasetTester(unittest.TestCase):
    """
    Unittest the clean Asos dataset.
    """
    def setUp(self):
        self.datadir = join(UNITTESTS_DIR, 'data/asos')
        self.dataset = AsosDataset(self.datadir)

    def test_dataset_length(self):
        self.assertEqual(len(self.dataset), 1)

    def test_getitem(self):
        datapoint = next(iter(self.dataset))

        app_img = datapoint['app_img']
        pose_img = datapoint['pose_img']
        target_img = datapoint['target_img']

        self.assertEqual(app_img.shape, [3, 256, 256])
        self.assertEqual(pose_img.shape, [3, 256, 256])
        self.assertEqual(target_img.shape, [3, 256, 256])

        # Check the data is between 0-1 and not 0-256
        self.assertTrue(app_img[0, 0, 0] <= 1)
        self.assertTrue(app_img[0, 0, 0] >= 0)
        self.assertTrue(pose_img[0, 0, 0] <= 1)
        self.assertTrue(pose_img[0, 0, 0] >= 0)
        self.assertTrue(target_img[0, 0, 0] <= 1)
        self.assertTrue(target_img[0, 0, 0] >= 0)
