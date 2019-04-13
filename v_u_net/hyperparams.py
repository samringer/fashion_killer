# Training params
overtrain = False
use_cuda = True
use_fp16 = True
KL_weight = 8e-6
bs = 4
num_epochs = 30
learning_rate = 1e-4

# Logging
ts_log_interval = 30

# Architecture
leakiness = 0.2
feature_weights = [1, 1, 1, 1, 1, 1]

# General
image_edge_size = 256
joint_crop_box_edge_size = 64

# Checkpoints
checkpoint_load_path = "/home/sam/experiments/V_U_Net/BIG_no_normalisation_correct_log/models/10000.chk"

checkpoint_interval = 5000

# Which joints we want to localise and feed into appearance encoder
from pose_drawer.pose_settings import JointType
joints_to_localise = [JointType.Nose,
                      JointType.RightShoulder,
                      JointType.LeftShoulder,
                      JointType.RightElbow,
                      JointType.LeftElbow,
                      JointType.RightHand,
                      JointType.LeftHand]
