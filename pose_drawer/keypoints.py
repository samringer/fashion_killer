from copy import deepcopy
from random import random


class KeyPoints():
    """
    An object that handles the state of a collection of
    pose keypoints.
    's' is a keypoint in stationary state
    'm' is a keypoint in moving state
    """
    def __init__(self):
        self.threshold = 1.5
        self.keypoints = self.get_null_keypoints()

        # Parameters for Markov model
        # p_m is prob moving from 'stationary' state to 'moving' state
        # p_t is prob moving from one 'moving' state to another
        self.alpha_1 = 2  # Used for calculating p_m

    def update_markov_model(self, model_out):
        """
        This is where the Markov model happens
        """
        new_keypoints = self.parse_keypoints_from_model_out(model_out)
        if self.keypoints == self.get_null_keypoints():
            self.keypoints = new_keypoints
            return

        for i, (old_kp, new_kp) in enumerate(zip(self.keypoints, new_keypoints)):
            #if old_kp['state'] == 's': # Keypoint is currently stationary
            dist = euc_distance(old_kp, new_kp)
            p_m = dist / (dist + self.alpha_1)
            if random() < p_m:
                self.keypoints[i] = new_kp

    # TODO: This is a hack and really needs sorting
    def __iter__(self):
        return iter(self.viable_coords)

    def parse_keypoints_from_model_out(self, pose_model_out):
        """
        Used for preparing the keypoints the are the output of
        the pretrained torchvision model.
        """
        model_kp = pose_model_out[0]['keypoints'].cpu().numpy()
        if not model_kp.size:
            return self.get_null_keypoints()
        scores = pose_model_out[0]['keypoints_scores'].cpu().numpy()[0]

        # Pose detector outputs in 800x800 coordinated but we want 256x256
        keypoints = [{'coord': tuple([256*j/800 for j in kp[:2]]),
                      'score': scores[i],
                      'state': 'm'} for i, kp in enumerate(model_kp[0])]

        keypoints = self.add_neck_keypoint(keypoints)
        return keypoints

    @staticmethod
    def get_null_keypoints():
        """
        Returns a null keypoints to use as placeholder.
        -10 is arbitrary number to represent a bad score.
        """
        return [{'coord': (0, 0), 'score': -10, 'state': 'm'} for _ in range(18)]

    @staticmethod
    def add_neck_keypoint(keypoints):
        """
        PyTorch pose detector does not return a neck keypoint.
        Add it in as an average of the left and right shoulders
        if both are found, (0, 0) otherwise.
        Args:
            keypoints (l o dicts)
        Returns:
            keypoints (l o dicts): Keypoints with neck added in.
        """
        r_shoulder_kp = keypoints[5]
        l_shoulder_kp = keypoints[6]

        neck_kp_x = (r_shoulder_kp['coord'][0] + l_shoulder_kp['coord'][0])//2
        neck_kp_y = (r_shoulder_kp['coord'][1] + l_shoulder_kp['coord'][1])//2
        neck_kp_score = (r_shoulder_kp['score'] + l_shoulder_kp['score'])/2
        neck_kp = {'coord': (neck_kp_x, neck_kp_y),
                   'score': neck_kp_score,
                   'state': 'm'}

        keypoints.insert(1, neck_kp)
        return keypoints

    @property
    def viable_coords(self):
        """
        Extract the coordinates that are over a certain score threshold
        for drawing. Uses coord (0, 0) otherwise.
        Returns:
            viable_coords (l o tuples): The viable keyponts for drawing.
        """
        viable_coords = []
        for keypoint in self.keypoints:
            if keypoint['score'] > self.threshold:
                viable_coords.append(keypoint['coord'])
            else:
                viable_coords.append((0, 0))
        return viable_coords

    @staticmethod
    def get_mirror_image_keypoints(keypoints):
        mirrored_keypoints = deepcopy(keypoints)
        mirrored_keypoints[2] = keypoints[3]
        mirrored_keypoints[3] = keypoints[2]

        mirrored_keypoints[4] = keypoints[5]
        mirrored_keypoints[5] = keypoints[4]

        mirrored_keypoints[6] = keypoints[7]
        mirrored_keypoints[7] = keypoints[6]

        mirrored_keypoints[8] = keypoints[9]
        mirrored_keypoints[9] = keypoints[8]

        mirrored_keypoints[10] = keypoints[11]
        mirrored_keypoints[11] = keypoints[10]

        mirrored_keypoints[12] = keypoints[13]
        mirrored_keypoints[13] = keypoints[12]

        mirrored_keypoints[14] = keypoints[15]
        mirrored_keypoints[15] = keypoints[14]

        mirrored_keypoints[16] = keypoints[17]
        mirrored_keypoints[17] = keypoints[16]
        return mirrored_keypoints

def euc_distance(kp_0, kp_1):
    """
    Returns Euclidean distance between two keypoints.
    Args:
        kp_0 (dict)
        kp_1 (dict)
    """
    x_dist = kp_1['coord'][0] - kp_0['coord'][0]
    y_dist = kp_1['coord'][1] - kp_0['coord'][1]
    return pow(x_dist**2 + y_dist**2, 0.5)
