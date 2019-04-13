import unittest
from os.path import join, dirname, realpath

from torch import optim
from torch.utils.data import DataLoader

from v_u_net.data_modules.dataset import VUNetDataset
from v_u_net.model.V_U_Net import VUNet
from v_u_net.train import _training_step

UNITTESTS_DIRECTORY = dirname(realpath(__file__))


class TestVUNetTrain(unittest.TestCase):
    """
    Test the ability of the VUNet train code to run a complete forward
    pass.
    """

    def test_train(self):
        datadir = join(UNITTESTS_DIRECTORY, 'data/v_u_net_data')
        dataset = VUNetDataset(datadir)
        dataloader = DataLoader(dataset, batch_size=1)
        model = VUNet()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        batch = next(iter(dataloader))
        output = _training_step(batch, model, optimizer)
        gen_img, total_loss, l1_loss, kl_divergence = output

        # TODO: 
        # Check the correct dimensionality of the generated img

        # Check the losses are all positive and a single number
        for loss in [total_loss, l1_loss, kl_divergence]:
            self.assertTrue(loss.item() > 0)
            self.assertEqual(loss.shape, [1])


if __name__ == "__main__":
    unittest.main()
