from enum import IntEnum


class PoseSettings():
    """
    Defines which connections/joints should appear in pose image
    and what color they should be.
    DOES NOT do any drawing.
    """

    def __init__(self):

        connection_colors_orig = [
            [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255],
            [0, 85, 255], [255, 0, 0], [0, 70, 255], [255, 170, 0], [255, 255, 0.],
            [255, 0, 85], [170, 255, 0], [85, 255, 0], [170, 0, 255.], [0, 0, 255],
            [0, 0, 255], [255, 0, 255], [170, 0, 255], [255, 0, 170]]
        joint_colors_orig = [
            [0, 0, 255], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
            [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
            [255, 0, 0], [0, 85, 255], [0, 0, 255], [85, 255, 255], [170, 0, 255],
            [255, 0, 255], [255, 0, 170], [255, 0, 85]]

        self.connection_colors = [[i/256 for i in j]
                                  for j in connection_colors_orig]
        self.joint_colors = [[i/256 for i in j]
                             for j in joint_colors_orig]

        self.keypoint_from_heatmap_threshold = 0.3

        self.desired_connections = [
            frozenset([JointType.Neck, JointType.RightWaist]),
            frozenset([JointType.RightWaist, JointType.RightKnee]),
            frozenset([JointType.RightKnee, JointType.RightFoot]),
            frozenset([JointType.Neck, JointType.LeftWaist]),
            frozenset([JointType.LeftWaist, JointType.LeftKnee]),
            frozenset([JointType.LeftKnee, JointType.LeftFoot]),
            frozenset([JointType.Neck, JointType.RightShoulder]),
            frozenset([JointType.RightShoulder, JointType.RightElbow]),
            frozenset([JointType.RightElbow, JointType.RightHand]),
            frozenset([JointType.Neck, JointType.LeftShoulder]),
            frozenset([JointType.LeftShoulder, JointType.LeftElbow]),
            frozenset([JointType.LeftElbow, JointType.LeftHand]),
            frozenset([JointType.Neck, JointType.Nose]),
            frozenset([JointType.Nose, JointType.RightEye]),
            frozenset([JointType.Nose, JointType.LeftEye]),
            frozenset([JointType.RightEye, JointType.RightEar]),
            frozenset([JointType.LeftEye, JointType.LeftEar])
        ]


class JointType(IntEnum):
    Nose = 0
    Neck = 1
    LeftEye = 2
    RightEye = 3
    LeftEar = 4
    RightEar = 5
    LeftShoulder = 6
    RightShoulder = 7
    LeftElbow = 8
    RightElbow = 9
    LeftHand = 10
    RightHand = 11
    LeftWaist = 12
    RightWaist = 13
    LeftKnee = 14
    RightKnee = 15
    LeftFoot = 16
    RightFoot = 17
