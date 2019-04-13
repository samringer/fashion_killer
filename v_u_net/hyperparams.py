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
