import torch
from torch import nn

from pose_detector.model.modules import Limb_Block, Joint_Block
from pose_detector.model.custom_VGG19 import get_custom_VGG19
import pose_detector.hyperparams as hp


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.VGG = get_custom_VGG19()

        self.limb_block_1 = Limb_Block(256)
        self.limb_block_2 = Limb_Block(256+hp.num_limbs)
        self.limb_block_3 = Limb_Block(256+hp.num_limbs)

        self.joint_block_1 = Joint_Block(256+hp.num_limbs)

    def forward(self, x):
        F = self.VGG(x)
        limb_map_1 = self.limb_block_1(F)

        x = torch.cat((F, limb_map_1), dim=1)
        limb_map_2 = self.limb_block_2(x)

        x = torch.cat((F, limb_map_2), dim=1)
        limb_map_3 = self.limb_block_3(x)

        x = torch.cat((F, limb_map_3), dim=1)
        joint_map_1 = self.joint_block_1(x)
        # TODO: if we add more than one joint blocks with the limb blocks
        # then we need to follow the catting scheme in the paper.

        part_affinity_fields = [limb_map_1, limb_map_2, limb_map_3]
        heat_maps = [joint_map_1]
        return part_affinity_fields, heat_maps
