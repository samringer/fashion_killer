from torch import nn

from Fashion_Killer.Model.U_Net import U_Net
from Fashion_Killer.Model.Appearance_Encoder import Appearance_Encoder

class V_U_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.appearance_encoder = Appearance_Encoder()
        self.u_net = U_Net()

    def forward(self, orig_img, orig_pose_img, target_pose_img, localised_joints):
        app_vec_1x1, app_vec_2x2, app_mu_1x1, app_mu_2x2 = self.appearance_encoder(orig_img, orig_pose_img, localised_joints)
        gen_img, pose_mu_1x1, pose_mu_2x2 = self.u_net(target_pose_img, app_vec_1x1, app_vec_2x2)
        gen_img = nn.Sigmoid()(gen_img)
        return gen_img, app_mu_1x1, app_mu_2x2, pose_mu_1x1, pose_mu_2x2
