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
        part_affinity_fields = batch['part_affinity_fields']
        kp_loss_mask = batch['kp_loss_mask']
        p_a_f_loss_mask = batch['p_a_f_loss_mask']

        self.assertEqual(img.shape, (self.batch_size, 3, 256, 256))

        self.assertEqual(list(keypoint_heat_maps.shape),
                         [self.batch_size, 18, 256, 256])
        self.assertEqual(list(part_affinity_fields.shape),
                         [self.batch_size, 17*2, 256, 256])
        self.assertEqual(list(kp_loss_mask.shape), [self.batch_size, 18])
        self.assertEqual(list(p_a_f_loss_mask.shape),
                         [self.batch_size, 17])


if __name__ == "__main__":
    unittest.main()
