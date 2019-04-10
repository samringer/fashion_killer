from torch.utils.data import DataLoader

from v_u_net.data_modules.dataset import V_U_Net_Dataset


def V_U_Net_DataLoader(batch_size, overtrain=False,
                       root_data_path='/home/sam/data/deepfashion'):
    """
    The dataloader used to prepare batches for training the V-U-Net
    Args:
        batch_size (int)
    """
    dataset = V_U_Net_Dataset(root_data_path, overtrain=overtrain)
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=True, num_workers=4)
