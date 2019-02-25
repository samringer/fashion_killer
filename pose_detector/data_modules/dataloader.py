from pathlib import Path
import numpy as np
import json
import cv2

from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from pose_detector.data_modules.dataset import Pose_Detector_Dataset


class Pose_Detector_DataLoader(DataLoader):

    def __init__(self, batch_size, root_data_path='/home/sam/data/COCO',
                 num_workers=4, overtrain=False,
                 min_joints_to_train_on=6):
        """
        Args:
            batch_size (int)
            root_data_path (str): Path to data.
            num_workers (int)
            overtrain (bool): True means same datapoint will always be
                              yielded.
        """
        dataset = Pose_Detector_Dataset(root_data_path,
                                        overtrain=overtrain,
                                        min_joints_to_train_on=min_joints_to_train_on)
        super().__init__(dataset, batch_size=batch_size,
                         shuffle=True, num_workers=num_workers)
