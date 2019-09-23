import torch
from torch import nn

from app_transfer.model.generator import Generator


class CachedGenerator(Generator):
    def load_cache(self, app_img, app_pose_img):
        self.app_enc_cache = self.forward_app_enc(app_img)
        self.app_pose_enc_cache = self.forward_app_pose_enc(app_pose_img)

    def forward_app_enc(self, app_img):
        with torch.no_grad():
            app_enc_0 = self.app_enc_conv_0(app_img)
            app_enc_1 = self.app_enc_conv_1(app_enc_0)
            app_enc_2 = self.app_enc_conv_2(app_enc_1)
            app_enc_3 = self.app_enc_conv_3(app_enc_2)
            app_enc_4 = self.app_enc_conv_4(app_enc_3)
            app_enc_5 = self.app_enc_conv_5(app_enc_4)
            app_enc_6 = self.app_enc_conv_6(app_enc_5)
            #app_enc_7 = self.app_enc_conv_7(app_enc_6)

        out = [app_enc_0, app_enc_1, app_enc_2, app_enc_3,
               app_enc_4, app_enc_5, app_enc_6] #, app_enc_7]
        return out

    def forward_app_pose_enc(self, app_pose_img):
        with torch.no_grad():
            app_pose_enc_0 = self.app_pose_enc_conv_0(app_pose_img)
            app_pose_enc_1 = self.app_pose_enc_conv_1(app_pose_enc_0)
            app_pose_enc_2 = self.app_pose_enc_conv_2(app_pose_enc_1)
            app_pose_enc_3 = self.app_pose_enc_conv_3(app_pose_enc_2)
            app_pose_enc_4 = self.app_pose_enc_conv_4(app_pose_enc_3)
            app_pose_enc_5 = self.app_pose_enc_conv_5(app_pose_enc_4)
            app_pose_enc_6 = self.app_pose_enc_conv_6(app_pose_enc_5)
            #app_pose_enc_7 = self.app_pose_enc_conv_7(app_pose_enc_6)

        out = [app_pose_enc_0, app_pose_enc_1, app_pose_enc_2,
               app_pose_enc_3, app_pose_enc_4, app_pose_enc_5,
               app_pose_enc_6] #, app_pose_enc_7]
        return out

    def forward(self, pose_img):
        pose_enc_0 = self.pose_enc_conv_0(pose_img)
        pose_enc_1 = self.pose_enc_conv_1(pose_enc_0)
        pose_enc_2 = self.pose_enc_conv_2(pose_enc_1)
        pose_enc_3 = self.pose_enc_conv_3(pose_enc_2)
        pose_enc_4 = self.pose_enc_conv_4(pose_enc_3)
        pose_enc_5 = self.pose_enc_conv_5(pose_enc_4)
        pose_enc_6 = self.pose_enc_conv_6(pose_enc_5)
        #pose_enc_7 = self.pose_enc_conv_7(pose_enc_6)

        # 128x128
        x = torch.cat([self.app_enc_cache[6], self.app_pose_enc_cache[6],
                       pose_enc_6], dim=1)
        x = self.dec_conv_0(self.app_enc_cache[5],
                            self.app_pose_enc_cache[5], pose_enc_5, x)
        x = self.dec_conv_1(self.app_enc_cache[4],
                            self.app_pose_enc_cache[4], pose_enc_4, x)

        # 256x256
        #x = torch.cat([self.app_enc_cache[7], self.app_pose_enc_cache[7],
        #               pose_enc_7], dim=1)
        #x = self.dec_conv_0(self.app_enc_cache[6],
        #                    self.app_pose_enc_cache[6], pose_enc_6, x)
        #x = self.dec_conv_1(self.app_enc_cache[5],
        #                    self.app_pose_enc_cache[5], pose_enc_5, x)
        #x = self.dec_conv_2(self.app_enc_cache[4],
        #                    self.app_pose_enc_cache[4], pose_enc_4, x)

        x = self.dec_conv_3(self.app_enc_cache[3],
                            self.app_pose_enc_cache[3], pose_enc_3, x)
        x = self.dec_conv_4(self.app_enc_cache[2],
                            self.app_pose_enc_cache[2], pose_enc_2, x)
        x = self.dec_conv_5(x)
        x = self.dec_conv_6(x)
        x = self.dec_conv_7(x)
        x = self.dec_conv_8(x)
        x = nn.Sigmoid()(x)
        return x
