import torch
from torch import nn

from pose_detector.model.modules import LimbBlock, JointBlock
from pose_detector.model.custom_VGG19 import get_custom_VGG19

num_joints = 18
num_limbs = 17


class PoseDetector(nn.Module):

    def __init__(self):
        super().__init__()
        self.VGG = get_custom_VGG19()

        self.limb_block_1 = LimbBlock(256)
        self.limb_block_2 = LimbBlock(256+2*num_limbs)
        self.limb_block_3 = LimbBlock(256+2*num_limbs)
        self.limb_block_4 = LimbBlock(256+2*num_limbs)

        self.joint_block_1 = JointBlock(256+2*num_limbs)
        self.joint_block_2 = JointBlock(256+2*num_limbs+num_joints)

    def forward(self, x):
        F = self.VGG(x)
        limb_map_1 = self.limb_block_1(F)

        x = torch.cat((F, limb_map_1), dim=1)
        limb_map_2 = self.limb_block_2(x)

        x = torch.cat((F, limb_map_2), dim=1)
        limb_map_3 = self.limb_block_3(x)

        x = torch.cat((F, limb_map_3), dim=1)
        limb_map_4 = self.limb_block_4(x)

        x = torch.cat((F, limb_map_4), dim=1)
        joint_map_1 = self.joint_block_1(x)

        x = torch.cat((F, limb_map_4, joint_map_1), dim=1)
        joint_map_2 = self.joint_block_2(x)

        part_affinity_fields = [limb_map_1, limb_map_2, limb_map_3,
                                limb_map_4]
        heat_maps = [joint_map_1, joint_map_2]
        return part_affinity_fields, heat_maps
