import cv2
import numpy as np

from pose_drawer.pose_settings import Pose_Settings


class Pose_Drawer():

    def __init__(self, canvas_size=256):
        """
        Args:
            canvas_size (int): Size of canvas edge in pixels.
        """
        self.pose_settings = Pose_Settings()
        self.canvas_size = int(canvas_size)

    def draw_pose_from_keypoints(self, joint_positions):
        """
        Args:
            joint_positions (l of tuples): Pixel positions of each joint.
            edge_size (int): Pixel edge size of canvas to draw image on.
        Returns:
            canvas (np array): np array of image with pose drawn on it.
        """
        canvas = np.zeros([self.canvas_size, self.canvas_size, 3])
        canvas = self._draw_limbs(canvas, joint_positions)
        canvas = self._draw_joints(canvas, joint_positions)
        return canvas

    def draw_pose_from_heatmaps(self, heat_maps):
        """
        pose_detector model outputs a heatmap for each joint prediction.
        This method draws a pose from these heatmaps.
        Args:
            heat_maps (l of np arrays): Joint confidence heat_maps.
                                        Must be np arrays, not PyTorch tensors.
        Returns:
            canvas (np array): np array of image with pose drawn on it.
        """
        keypoints = self.extract_keypoints_from_heatmaps(heat_maps)
        return self.draw_pose_from_keypoints(keypoints)

    def extract_keypoints_from_heatmaps(self, heat_maps):
        """
        pose_detector model outputs a heatmap for each joint prediction.
        This method extracts the keypoints from these heatmaps.
        Args:
            heat_maps (l of np arrays): Joint confidence heat_maps.
                                        Must be np arrays, not PyTorch tensors.
        Returns:
            keypoints (l of tuples): The positions of the keypoints (if
                                     they are found else [0, 0])
        """
        keypoints = []
        threshold = self.pose_settings.keypoint_from_heatmap_threshold
        for heat_map in heat_maps:
            max_heat = np.amax(heat_map)
            if max_heat >= threshold:
                keypoint = np.array(np.unravel_index(heat_map.argmax(), heat_map.shape))
                keypoint = np.array([keypoint[1], keypoint[0]])  # Unravelling flips dims
            else:
                keypoint = np.array([0, 0])
            keypoints.append(keypoint)
        return keypoints

    def _draw_limbs(self, canvas, joint_positions):
        desired_connections = self.pose_settings.desired_connections
        connection_colors = self.pose_settings.connection_colors
        for connection, connection_color in zip(desired_connections, connection_colors):
            start_joint, end_joint = connection
            start_point, end_point = joint_positions[start_joint.value], joint_positions[end_joint.value]
            # Only draw connection if start point and end point both found.
            if all([self._point_found(point.tolist()) for point in [start_point, end_point]]):
                canvas = draw_line_on_canvas(canvas, start_point, end_point, connection_color)
        return canvas

    def _draw_joints(self, canvas, joint_positions):
        joint_colors = self.pose_settings.joint_colors
        for joint_position, joint_color in zip(joint_positions, joint_colors):
            # Only draw joint if joint found by OpenPose
            if self._point_found(joint_position.tolist()):
                canvas = draw_point_on_canvas(canvas, joint_position, joint_color)
        return canvas

    def _point_found(self, point):
        """
        Point has value (0, 0) if point not found.
        Args:
            point (list): List containing keypoint coordinates of point.
        Returns:
            found (bool): Whether the point in question was found by the pose detector.
        """
        return point != [0, 0] and point != [-self.canvas_size, -self.canvas_size]


def draw_line_on_canvas(canvas, start_point, end_point, color, thickness=3):
    """
    Args:
        canvas (np array): The canvas you want to draw the limbs on.
        start_point (tuple): x, y coordinates (in pixels) to start the line.
        end_point (tuple): x, y coordinates (in pixels) to end the line.
        color (list): List of RBG color to draw the line in.
        thickness (int): Line thickness in pixels
    Returns:
        canvas (np array): The canvas with the line drawn on it
    """
    color = (color[2], color[1], color[0])  # cv2 expects color in BGR format
    start_point = tuple(int(i) for i in start_point)  # cv2 expects integers
    end_point = tuple(int(i) for i in end_point)

    canvas = cv2.line(canvas, start_point, end_point, color, thickness)
    return canvas


def draw_point_on_canvas(canvas, point, color, thickness=5):
    """
    Args:
        canvas (np array): The canvas you want to draw the limbs on.
        start_point (tuple): x, y coordinates (in pixels) of position of point.
        color (list): List of RBG color to draw the point in.
        thickness (int): Line thickness in pixels.
    Returns:
        canvas (np array): The canvas with the line drawn on it
    """
    color = (color[2], color[1], color[0])  # cv2 expects color in BGR format
    point = tuple(map(int, point))  # cv2 expects integers
    canvas = cv2.circle(canvas, point, thickness, color, -1) # -1 to draw a filled circle
    return canvas
