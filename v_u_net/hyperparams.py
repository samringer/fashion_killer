# Training params
over_train = False
use_cuda = True
use_fp16 = True
KL_weight = 8e-6
bs = 8
num_epochs = 30
learning_rate = 0.0001
beta_1 = 0.0
beta_2 = 0.9

# Inference
use_cuda_inference = True

# Logging
ts_log_interval = 30

# Architecture
leakiness = 0.2
feature_weights = [1, 1, 1, 1, 1, 1]

# General
image_edge_size = 256
joint_crop_box_edge_size = 64

# Which joints we want to localise and feed into appearance encoder
from pose_drawer.pose_settings import JointType
joints_to_localise = [JointType.Nose,
                      JointType.RightShoulder,
                      JointType.LeftShoulder,
                      JointType.RightElbow,
                      JointType.LeftElbow,
                      JointType.RightHand,
                      JointType.LeftHand]
