from torch import nn

from pose_detector.model.modules import Joint_Map_Block
from pose_detector.model.custom_VGG19 import get_custom_VGG19
import pose_detector.hyperparams as hp


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.VGG = get_custom_VGG19()
        self.joint_map_block_1 = Joint_Map_Block(256)
        self.joint_map_block_2 = Joint_Map_Block(256+hp.num_joints)
        self.joint_map_block_3 = Joint_Map_Block(256+2*hp.num_joints)
        self.joint_map_block_4 = Joint_Map_Block(256+3*hp.num_joints)

    def forward(self, x):
        x = self.VGG(x)
        x, heat_maps_1 = self.joint_map_block_1(x)
        x, heat_maps_2 = self.joint_map_block_2(x)
        x, heat_maps_3 = self.joint_map_block_3(x)
        _, heat_maps_4 = self.joint_map_block_4(x)
        heat_maps = [heat_maps_2, heat_maps_3, heat_maps_4]
        return heat_maps
