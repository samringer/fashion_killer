import numpy as np
from skimage.transform import resize

import v_u_net.hyperparams as hp


def get_localised_joints(img, joints_to_localise, joint_positions):
    """
    Generate multiple crops around specific joints which are resized to
    be the same size as the original image.
    Helps appearance encoder localise appearance to specific points in image.
    These crops are then fed into the appearance encoder.
    Args:
        orig_img (np array): Original img to localise joints from.
        joints_to_localise (list): List of ids of the joints to localise.
        joint_positions (list): List of pixel positions of all the joints
    Returns:
        localised_joints (list): np arrays of each localised joint image.
    """
    localised_joints = []
    for joint_id in joints_to_localise:
        joint_position = joint_positions[joint_id]

        # Black image used if joint has not been found by pose detector.
        if not _point_found(joint_position.tolist()):
            joint_img = np.zeros(img.shape)
        else:
            joint_img = get_joint_image(img, joint_position,
                                        hp.joint_crop_box_edge_size)
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
    upsampled_joint_img = resize(joint_img,
                                 (orig_img_width, orig_img_height),
                                 mode='reflect',
                                 anti_aliasing=True)
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
    half = edge_size//2
    x, y = centre_point
    padded_img = np.pad(img,
                        ((half, half), (half, half), (0, 0)),
                        'constant')

    # Need to account for extra padding
    x += half
    y += half
    box = padded_img[y-half:y+half, x-half:x+half, :]
    return box


def _point_found(point):
    """
    OpenPose returns [-1, -1] or [0, 0] if point not found.
    Args:
        point (list): List containing OpenPose coordinates of point
    Returns:
        found (bool): Whether the point in question was found by OpenPose
    """
    return point not in ([-hp.image_edge_size, -hp.image_edge_size],
                         [0, 0])
