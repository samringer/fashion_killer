from torch import nn

from v_u_net.model.u_net import UNet
from v_u_net.model.appearance_encoder import AppearanceEncoder


class VUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.appearance_encoder = AppearanceEncoder()
        self.u_net = UNet()

    def forward(self, orig_img, orig_pose_img, target_pose_img, localised_joints):
        appearance = self.appearance_encoder(orig_img, orig_pose_img,
                                             localised_joints)
        app_vec_1x1, app_vec_2x2, app_mu_1x1, app_mu_2x2 = appearance
        gen_img, pose_mu_1x1, pose_mu_2x2 = self.u_net(target_pose_img,
                                                       app_vec_1x1,
                                                       app_vec_2x2)
        gen_img = nn.Sigmoid()(gen_img)
        return gen_img, app_mu_1x1, app_mu_2x2, pose_mu_1x1, pose_mu_2x2


class CachedVUNet(VUNet):
    """
    Used to speed things up by caching the appearance vector.
    Used only for real time inference. The appearance encodings
    are provided externally.
    """
    def __init__(self):
        super().__init__()

    def gen_app_cache(self, orig_img, orig_pose_img, localised_joints):
        cache = self.appearance_encoder(orig_img, orig_pose_img,
                                        localised_joints)
        self.app_vec_1x1, self.app_vec_2x2, _, _ = cache

    def forward(self, target_pose, app_vec_1, app_vec_2):
        gen_img, _, _ = self.u_net(target_pose, app_vec_1, app_vec_2)
        gen_img = nn.Sigmoid()(gen_img)
        return gen_img
