from torch.utils.data import DataLoader

from v_u_net.data_modules.dataset import VUNetDataset


def VUNetDataLoader(batch_size, overtrain=False,
                    root_data_path='/home/sam/data/deepfashion',
                    num_workers=4):
    """
    The dataloader used to prepare batches for training the V-U-Net
    Args:
        batch_size (int)
    """
    dataset = VUNetDataset(root_data_path, overtrain=overtrain)
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=True, num_workers=num_workers,
                      pin_memory=True)
