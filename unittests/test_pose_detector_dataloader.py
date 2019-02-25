import unittest
from os.path import join, dirname, realpath

from torch.utils.data import DataLoader

from pose_detector.data_modules.dataloader import Pose_Detector_DataLoader


UNITTESTS_DIRECTORY = dirname(realpath(__file__))


class DataLoader_Unittests(unittest.TestCase):

    def setUp(self):
        self.datadir = join(UNITTESTS_DIRECTORY, 'data/pose_detector_data')
        self.batch_size = 1
        self.dataLoader = Pose_Detector_DataLoader(self.batch_size,
                                                   self.datadir,
                                                   num_workers=0)

    def test_get_batch(self):
        """
        Check the contents of a batch are as expected.
        Similar to test_getitem when testing dataset.
        """
        batch = next(iter(self.dataLoader))

        img = batch['img']
        keypoint_heat_maps = batch['keypoint_heat_maps']
        loss_mask = batch['loss_mask']

        desired_shape = (self.batch_size, 3, 256, 256)
        self.assertEqual(img.shape, desired_shape)
        self.assertEqual(list(keypoint_heat_maps.shape), [self.batch_size, 18, 256, 256])
        self.assertEqual(list(loss_mask.shape), [self.batch_size, 18])


if __name__ == "__main__":
    unittest.main()
