import json
import cv2
import random
import math
import numpy as np
from os.path import join

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image, ImageDraw

from pose_drawer.pose_settings import PoseSettings


class PoseDetectorDataset(Dataset):
    """
    Dataset consists of images with associated keypoints.
    """
    def __init__(self, root_data_dir='/home/sam/data/COCO',
                 overtrain=False,
                 min_joints_to_train_on=6):

        annotations_path = join(root_data_dir, 'annotations')
        keypoints_path = join(annotations_path, 'keypoints_train.json')
        self.imgs_path = join(root_data_dir, 'train_imgs')

        self.max_dim = 256
        self.overtrain = overtrain
        self.min_joints_to_train_on = min_joints_to_train_on

        # Needed for drawing the Part Affinity Fields.
        self.pose_settings = PoseSettings()

        with open(str(keypoints_path), 'r') as in_f:
            self.imgs_data = json.load(in_f)['annotations']

        self.valid_img_indicies = _get_valid_imgs(self.imgs_data,
                                                  self.min_joints_to_train_on)

    def __len__(self):
        return len(self.valid_img_indicies)

    def __getitem__(self, index):

        if self.overtrain:
            index = 0  # Use 12 for a 'good' pose

        # Perform mapping from all indices to valid indices
        valid_index = self.valid_img_indicies[index]
        img_data = self.imgs_data[valid_index]
        img_id = str(img_data['image_id']).zfill(12)

        img_file_path = join(self.imgs_path, img_id+'.jpg')
        img = cv2.imread(img_file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = np.asarray(img) / 256
        img, keypoint_trans_info = self._prepare_img(img, img_data)
        img = transforms.ToTensor()(img)

        keypoints = _extract_keypoints_from_img_data(img_data)
        keypoints = _adjust_keypoints(keypoints, *keypoint_trans_info)

        kp_loss_mask = _get_kp_loss_mask(keypoints)
        keypoint_heat_maps = _create_heat_maps(keypoints, self.max_dim)

        p_a_fs, p_a_f_loss_mask = self._create_part_affinity_fields(keypoints)

        return {'img': img.float(),
                'keypoint_heat_maps': keypoint_heat_maps.float(),
                'kp_loss_mask': kp_loss_mask,
                'p_a_f': p_a_fs.float(),
                'p_a_f_loss_mask': p_a_f_loss_mask}

    def _prepare_img(self, img, img_data):
        """
        Prepare an image for input into model.
        Performs cropping, resizing and padding.
        Note this DOES NOT include PyTorch transformations like
        converting to a tensor.
        Args:
            img (np array)
            img_data (dict): Data about things like bbox in img.
        Returns:
            out_img (np array): Cropped image of correct square size.
            keypoint_trans_info (tuple):
                Info to correctly adjust keypoint location.
        """
        (x_0, y_0), w, h = _get_cropping_rectangle_from_img_data(img_data)
        cropped_img = img[y_0:y_0+h, x_0:x_0+w, :]
        resized_img, scale_factor = _resize_img(cropped_img, self.max_dim)

        height, width, _ = resized_img.shape
        x_offset = random.randint(0, self.max_dim - width)
        y_offset = random.randint(0, self.max_dim - height)

        if self.overtrain:
            x_offset, y_offset = 0, 0

        padded_img = _pad_img(resized_img, x_offset, y_offset)
        keypoint_trans_info = ((x_0, y_0), x_offset, y_offset, scale_factor)
        return padded_img, keypoint_trans_info

    def _create_part_affinity_fields(self, keypoints):
        """
        Create PAF for each limb.
        Returns:
            part_affinity_fields (tensor): A single tensor of all the PAFs.
            paf_loss_mask (tensor): Loss mask so non-present limbs are not
                                    scored during training.
        """
        desired_connections = self.pose_settings.desired_connections
        part_affinity_fields = []
        p_a_f_loss_mask = []

        for limb in desired_connections:
            start_joint, end_joint = limb
            start_point = keypoints[start_joint.value]
            end_point = keypoints[end_joint.value]

            p_a_f = np.zeros([2, self.max_dim, self.max_dim])
            if start_point != (0, 0) and end_point != (0, 0):
                p_a_f_loss_mask.extend([1., 1.])
                p_a_f = _draw_part_affinity_field(start_point,
                                                  end_point,
                                                  p_a_f)
            else:
                p_a_f_loss_mask.extend([0., 0.])

            part_affinity_fields.append(torch.from_numpy(p_a_f))
        part_affinity_fields = torch.cat(part_affinity_fields)
        return part_affinity_fields, torch.Tensor(p_a_f_loss_mask)


def _get_valid_imgs(data, min_joints_to_train_on=1):
    """
    Used to get indexes of all the images in the dataset
    that contain at least min_joints_to_train_on keypoints.
    Args:
        data (dict): Contains info about each img in dataset.
        min_joints_to_train_on (int): All images used must contain
                                      at least this many keypoints.
    Returns:
        valid_img_ids (l o int): Image ids that contain
                                 at least required num keypoints.
    """
    valid_img_indicies = []
    for index, img in enumerate(data):
        if img['num_keypoints'] >= min_joints_to_train_on:
            valid_img_indicies.append(index)
    return valid_img_indicies


def _get_kp_loss_mask(keypoints):
    """
    Keypoints that are not present in data should
    have corresponding losses masked during training.
    Args:
        keypoints (l of tuples): Pixel positions of all keypoints.
    Returns:
        kp_loss_mask (PyTorch tensor): 0 if joint not found, 1 if it is.
    """
    kp_loss_mask = []
    for point in keypoints:
        if point == (0, 0):
            # TODO: Check if this works
            kp_loss_mask.append(0.01)
        else:
            kp_loss_mask.append(1.)
    return torch.Tensor(kp_loss_mask)


def _extract_keypoints_from_img_data(data):
    """
    Parse an images data dictionary to extract keypoints.
    Keypoint data comes in triplets like (x, y, hidden) where
    hidden is a number representing if the keypoint is hidden.
    (This is ignored here).
    This also adds in a neck keypoint.
    Args:
        data (dict): Contains info about each img in dataset.
    Returns:
        keypoints (l o tuples)
    """
    keypoints = data['keypoints']
    num_keypoints = len(keypoints)//3  # As keypoints come in triplets.
    valid_keypoints = [(keypoints[i*3], keypoints[(i*3)+1])
                       for i in range(num_keypoints)]
    valid_keypoints = _add_neck_keypoint(valid_keypoints)
    return valid_keypoints


def _add_neck_keypoint(keypoints):
    """
    COCO by default does not contain a neck keypoint.
    This function adds it in as an average of the left
    and right shoulders if both are found, (0, 0) otherwise.
    Args:
        keypoints (l o tuples)
    Returns:
        keypoints (l o tuples): Keypoints with neck added in.
    """
    r_shoulder_keypoint = keypoints[5]
    l_shoulder_keypoint = keypoints[6]
    if r_shoulder_keypoint != (0, 0) and l_shoulder_keypoint != (0, 0):
        neck_keypoint_x = (r_shoulder_keypoint[0] + l_shoulder_keypoint[0])//2
        neck_keypoint_y = (r_shoulder_keypoint[1] + l_shoulder_keypoint[1])//2
        neck_keypoint = (neck_keypoint_x, neck_keypoint_y)
    else:
        neck_keypoint = (0, 0)
    keypoints.insert(1, neck_keypoint)
    return keypoints


def _get_cropping_rectangle_from_img_data(data):
    """
    Calc the rectangle used for cropping image around POI.
    Args:
        data (dict): Contains info about each img in dataset.
    Returns:
        ul_point (tuple): Pixel coordinates of upper left corner of rect.
        width (int): Pixel width of rect.
        height (int): Pixel height of rect.
    """
    x_1, y_1, width, height = data['bbox']
    ul_point = (int(x_1), int(y_1))
    return ul_point, int(width), int(height)


def _resize_img(img, max_dim):
    """
    Resize an image so longest dim is exactly max dim.
    Args:
        img (np array): np array representing img.
        max_dim (int): Desired max length of any img dimension.
    Returns:
        resized_img (np array): Resized img.
        scale_factor (float): Needed to correctly adjust keypoint
                              location.
    """
    img_width, img_height, _ = img.shape
    current_max_dim = max(img_width, img_height)
    scale_factor = max_dim / current_max_dim
    resized_img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
    return resized_img, scale_factor


def _pad_img(img, x_offset, y_offset):
    """
    Pad an image along its smallest dim such that
    it forms a square.
    Assumes image is RGB.
    Note that at least one of x_offset and y_offset will be zero.
    Args:
        img (np array): np array of input img.
        x_offset (int): num pixels to offset width padding by.
        y_offset (int): num pixels to offset width padding by.
    Returns:
        canvas (np array): A square padded img.
    """
    height, width, _ = img.shape
    max_dim = max(height, width)
    canvas = np.zeros([max_dim, max_dim, 3])
    canvas[y_offset:height+y_offset, x_offset:width+x_offset, :] = img
    return canvas


def _adjust_keypoints(keypoints, ul_point, x_offset, y_offset, scale_factor):
    """
    Adjust keypoints to account for transformations to the input
    image. We only want to adjsut keypoints that are not (0, 0).
    Args:
        keypoints (l o tuples): The original keypoints.
        ul_point (tuple of ints): Position of upperleft corner of bbox.
        x_offset (int): Num pixels to offset x coordinates by.
        y_offset (int): Num pixels to offset y coordinates by.
        scale_factor (float):
    Returns:
        adj_keypoints (l o tuples): The adjusted keypoints.
    """
    x_0, y_0 = ul_point
    adj_keypoints = []

    for (x, y) in keypoints:
        if (x, y) != (0, 0):  # Only adjust the keypoint if it is found.
            x = (x-x_0)*scale_factor
            y = (y-y_0)*scale_factor
            x = int(x+x_offset)
            y = int(y+y_offset)
        adj_keypoints.append((x, y))

    return adj_keypoints


def _create_heat_maps(keypoints, edge_size, sigma=20):
    """
    Create heat maps for a set of keypoints.
    Args:
        keypoints (l o tuples): Pixels positions of all keypoints.
        edge_size(int): Pixel size of edge of heatmap.
        sigma (float): Param that controls spread of heatmap peak.
    Returns:
        heat_maps (PyTorch tensor): A single tensor of all the heatmaps.
    """
    heat_maps = []
    for keypoint in keypoints:
        heat_map = _create_heat_map(keypoint, edge_size, sigma)
        heat_maps.append(torch.from_numpy(heat_map))
    heat_maps = torch.stack(heat_maps)
    return heat_maps


def _create_heat_map(keypoint, edge_size, sigma=20):
    """
    Create a heat map for a given keypoint.
    Args:
        keypoint (tuple of ints): Pixels positions of keypoint.
        edge_size(int): Pixel size of edge of heatmap.
        sigma (float): Param that controls spread of heatmap peak.
    Returns:
        heat_map (np array)
    """
    heat_map = np.zeros((edge_size, edge_size))
    if keypoint == (0, 0):
        return heat_map

    x, y = keypoint
    for row_num in range(edge_size):
        for col_num in range(edge_size):
            dist_from_keypoint = (row_num-x)**2 + (col_num-y)**2
            exponent = dist_from_keypoint/(sigma**2)
            heat = math.e**(-exponent)
            heat_map[col_num][row_num] = heat
    return heat_map


def _draw_part_affinity_field(start_point, end_point, canvas):
    """
    Creates a part affinity map for the limb between two joints.
    Args:
        start_point (tuple of ints): Pixel position of start
                                     joint of limb.
        end_point (tuple of ints): Pixel position of end
                                   joint of limb.
        canvas (np array): Canvas to draw PAF on.
    Returns:
        canvas (np array): Canvas with PAF drawn on it.
    """
    limb_pixel_width = 4

    limb_vec = np.subtract(end_point, start_point)
    limb_vec_length = np.linalg.norm(limb_vec)
    if limb_vec_length < 1e-4:
        return canvas
    unit_vec = np.divide(limb_vec, limb_vec_length)
    perp_unit_vec = np.array([unit_vec[1], -unit_vec[0]])
    perp_vec = perp_unit_vec * limb_pixel_width

    u_left = start_point + perp_vec
    l_left = start_point - perp_vec
    u_right = start_point + perp_vec + limb_vec
    l_right = start_point - perp_vec + limb_vec
    rect = [u_left, l_left, l_right, u_right, u_left]

    # We are drawing a 2d vector on the PAFs.
    # To interact with the draw API we need to split canvas in two.
    img_0 = Image.fromarray(canvas[0, :, :])
    img_1 = Image.fromarray(canvas[1, :, :])
    for i, img in enumerate([img_0, img_1]):
        draw = ImageDraw.Draw(img)
        draw.polygon([tuple(p) for p in rect], fill=1)

    img_0 *= unit_vec[0]
    img_1 *= unit_vec[1]
    canvas = np.stack([np.asarray(img_0), np.asarray(img_1)])
    canvas = canvas.astype(float)  # Numpy converts to double
    return canvas
