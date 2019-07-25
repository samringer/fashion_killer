import cv2
import numpy as np

from pose_drawer.pose_settings import PoseSettings


class PoseDrawer():

    def __init__(self, canvas_size=256):
        """
        Args:
            canvas_size (int): Size of canvas edge in pixels.
        """
        self.pose_settings = PoseSettings()
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
        desired_connections = self.pose_settings.desired_connections
        connection_colors = self.pose_settings.connection_colors
        joint_colors = self.pose_settings.joint_colors

        # Draw each limb one at a time
        for connection, connection_color in zip(desired_connections, connection_colors):
            start_joint, end_joint = connection
            start_point = joint_positions[start_joint.value]
            end_point = joint_positions[end_joint.value]
            # Only draw connection if start point and end point both found.
            if all([self._point_found(point) for point in [start_point, end_point]]):
                canvas = draw_line_on_canvas(canvas, start_point, end_point, connection_color)

                # Draw the joints (only if they are connected to a limb
                # This stops the drawing of any isolated dots.
                start_joint_color = joint_colors[start_joint.value]
                end_joint_color = joint_colors[end_joint.value]
                canvas = draw_point_on_canvas(canvas, start_point, start_joint_color)
                canvas = draw_point_on_canvas(canvas, end_point, end_joint_color)
        return canvas

    def _point_found(self, point):
        """
        Point has value (0, 0) if point not found.
        Args:
            point (list): List containing keypoint coordinates of point.
        Returns:
            found (bool): Point found by the pose detector.
        """
        if isinstance(point, np.ndarray):
            point = tuple(point.tolist())
        return point != (0, 0)


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
    # cv2 expects color in BGR format
    color = (color[2], color[1], color[0])
    point = tuple(map(int, point))  # cv2 expects integers
    canvas = cv2.circle(canvas, point, thickness, color, -1) # -1 to draw a filled circle
    return canvas
