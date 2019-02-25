import cv2
import numpy as np

import v_u_net.hyperparams as hp


def get_localised_joints(img, joints_to_localise, joint_positions):
    """
    Generate multiple crops around specific joints which are resized to
    be the same size as the original image.
    Helps appearance encoder localise appearance to specific points in image.
    These crops are then fed into the appearance encoder.
    Args:
        orig_img (PIL JPEG object): The original image to localise joints from.
        joints_to_localise (list): List of ids of the joints to localise.
        joint_positions (list): List of pixel positions of all the joints
    Returns:
        localised_joints (list): List of the np arrays of each localised joint image
    """
    img = np.array(img) / 256.
    localised_joints = []
    for joint_id in joints_to_localise:
        joint_position = joint_positions[joint_id]

        #  A black image is used if joint has not been found by pose detector.
        if not _point_found(joint_position.tolist()):
            joint_img = np.zeros(img.shape)
        else:
            joint_img = get_joint_image(img, joint_position, hp.joint_crop_box_edge_size)

        localised_joints.append(joint_img)
    return localised_joints


def get_joint_image(orig_img, centre_point, edge_size):
    """
    Crops a box around a specific point and resizes it to be the same size
    as the original image.
    Args:
        orig_img (np array): np array representing the image to crop of size HxWx3.
        centre_point ((int, int)): Tuple with coordinates of centre point in pixels.
        edge_size (int): Size of edge of cropped box in pixels.
    Returns:
        upsampled_joint_img (np array): Image of joint of same size as original image.
    """
    orig_img_width, orig_img_height, _ = orig_img.shape
    joint_img = _crop_image(orig_img, centre_point, edge_size)
    upsampled_joint_img = _upsample_image(joint_img, orig_img_width, orig_img_height)
    return upsampled_joint_img


def _crop_image(img, centre_point, edge_size):
    """
    Crop a box of size (edge_size x edge_size) out of an input image.
    Image is padded with 0s to ensure that a box of correct size is always returned.
    Args:
        img (np array): np array representing the image to crop of size HxWx3.
        centre_point ((int, int)): Tuple with coordinates of centre point in pixels.
        edge_size (int): Size of edge of cropped box in pixels.
    Returns:
        box (np array): np array of the croppped box.
    """
    half_size = edge_size//2
    x, y = centre_point
    padded_img = np.pad(img, ((half_size,half_size), (half_size,half_size), (0,0)), 'constant')

    # Need to account for extra padding
    x = x + half_size
    y = y + half_size
    box = padded_img[y-half_size:y+half_size, x-half_size:x+half_size, :]
    return box


def _upsample_image(img, desired_width, desired_height):
    """
    Upscales an input image to a desired size
    Args:
        img (np array): np array of image to resize.
        desired_width (int): Desired width of new image in pixels.
        desired_height (int): Desired height of new image in pixels.
    """
    return cv2.resize(img, (desired_width, desired_height))


def _point_found(point):
    """
    OpenPose returns [-1, -1] or [0, 0] if point not found.
    Args:
        point (list): List containing OpenPose coordinates of point
    Returns:
        found (bool): Whether the point in question was found by OpenPose
    """
    return point != [-hp.image_edge_size, -hp.image_edge_size] and point != [0, 0]
