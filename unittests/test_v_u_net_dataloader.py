import unittest
from os.path import join, dirname, realpath

from torch.utils.data import DataLoader

from v_u_net.data_modules.dataloader import V_U_Net_DataLoader


UNITTESTS_DIRECTORY = dirname(realpath(__file__))


class DataLoader_Unittests(unittest.TestCase):

    def setUp(self):
        self.datadir = join(UNITTESTS_DIRECTORY, 'data/v_u_net_data')
        self.batch_size = 1
        self.dataLoader = V_U_Net_DataLoader(self.batch_size,
                                             root_data_path=self.datadir)


    def test_get_batch(self):
        """
        Check the contents of a batch are as expected.
        Similar to test_getitem when testing dataset.
        """
        batch = next(iter(self.dataLoader))

        app_img = batch['app_img']
        pose_img = batch['pose_img']
        localised_joints = batch['localised_joints']

        app_img_shape = list(app_img.shape)
        self.assertEqual(app_img_shape, [1, 3, 256, 256])

        pose_img_shape = list(pose_img.shape)
        self.assertEqual(pose_img_shape, [1, 3, 256, 256])

        localised_joints_shape = list(localised_joints.shape)
        self.assertEqual(localised_joints_shape, [1, 21, 256, 256])


if __name__ == "__main__":
    unittest.main()
