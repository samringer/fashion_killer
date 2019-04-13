import unittest
from os.path import join, dirname, realpath

from torch import optim
from torch.utils.data import DataLoader

from pose_detector.data_modules.dataset import Pose_Detector_Dataset
from pose_detector.model.model import Model
from pose_detector.train import _training_step

FUNCTESTS_DIRECTORY = dirname(realpath(__file__))


class TestPoseDetectorTrain(unittest.TestCase):
    """
    Test the ability of the Pose Detector train code to run a complete
    forward and backward pass.
    """

    def test_train(self):
        datadir = join(FUNCTESTS_DIRECTORY, 'data/pose_detector')
        dataset = Pose_Detector_Dataset(datadir)
        dataloader = DataLoader(dataset, batch_size=1)
        model = Model()
        # TODO: 
        model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        batch = next(iter(dataloader))
        output = _training_step(batch, model, optimizer)
        pred_pafs, pred_heat_maps, loss = output

        # TODO: 
        # Check the correct dimensionality of the pafs img

        # TODO: 
        # Check the correct dimensionality of the heat map img

        # Check loss positive and a single number
        self.assertTrue(loss.item() > 0)
        self.assertEqual(loss.shape, [1])


if __name__ == "__main__":
    unittest.main()
