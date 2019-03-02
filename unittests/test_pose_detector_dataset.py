import unittest, pickle, json
import numpy as np
from os.path import join, dirname, realpath

import torch, cv2

from pose_detector.data_modules.dataset import Pose_Detector_Dataset

UNITTESTS_DIRECTORY = dirname(realpath(__file__))


class Dataset_Unittester(unittest.TestCase):
    """
    Unittest the dataset used for pose detection.
    """

    def setUp(self):
        self.datadir = join(UNITTESTS_DIRECTORY, 'data/pose_detector_data')
        self.dataset = Pose_Detector_Dataset(self.datadir)

    def test_valid_imgs(self):
        """
        Only use images that contain at least one keypoint.
        """
        self.assertEqual(self.dataset.valid_img_indicies, [0])

    def test_length_dataset(self):
        """
        Dataset should only contain valid imgs.
        """
        self.assertEqual(len(self.dataset), 1)

    def test_getitem(self):
        """
        Test that an image and other data returned from
        the dataset fits the correct dimensions.
        """
        datapoint = next(iter(self.dataset))
        img = datapoint['img']
        keypoint_heat_maps = datapoint['keypoint_heat_maps']
        part_affinity_fields = datapoint['part_affinity_fields']
        kp_loss_mask = datapoint['kp_loss_mask']
        p_a_f_loss_mask = datapoint['p_a_f_loss_mask']

        # Note PyTorch wants 3xHxW
        desired_shape = (3, self.dataset.max_dim, self.dataset.max_dim)
        self.assertEqual(img.shape, desired_shape)

        self.assertEqual(list(keypoint_heat_maps.shape),
                         [18, self.dataset.max_dim, self.dataset.max_dim])
        self.assertEqual(list(part_affinity_fields.shape),
                         [17, self.dataset.max_dim, self.dataset.max_dim,2])
        self.assertEqual(list(kp_loss_mask.shape), [18])
        self.assertEqual(list(p_a_f_loss_mask.shape), [17])

    def test_prepare_img(self):
        """
        An input image should be returned with the correct
        dimensions for training.
        """
        img_data = self.dataset.imgs_data[0]
        img_name = str(img_data['image_id']).zfill(12)
        img_path = join(self.dataset.imgs_path, img_name+'.jpg')
        max_dim = self.dataset.max_dim

        img = cv2.imread(img_path)
        img, _ = self.dataset._prepare_img(img, img_data)
        self.assertEqual(img.shape, (max_dim, max_dim, 3))

    def test_overtrain(self):
        """
        Check that the same batch is always returned when
        overtraining.
        """
        self.dataset.overtrain = True
        first_batch = next(iter(self.dataset))
        second_batch = next(iter(self.dataset))

        self.assertEqual(first_batch['img'].tolist(),
                         second_batch['img'].tolist())

        self.assertEqual(first_batch['keypoint_heat_maps'].tolist(),
                         second_batch['keypoint_heat_maps'].tolist())

        self.assertEqual(first_batch['kp_loss_mask'].tolist(),
                         second_batch['kp_loss_mask'].tolist())

        self.assertEqual(first_batch['p_a_f_loss_mask'].tolist(),
                         second_batch['p_a_f_loss_mask'].tolist())

    def test_adjust_keypoints(self):
        """
        Keypoints need to have there position adjusted to account
        for transformations to the input image.
        """
        from pose_detector.data_modules.dataset import _adjust_keypoints, _extract_keypoints_from_img_data
        img_data = self.dataset.imgs_data[0]
        original_keypoints = _extract_keypoints_from_img_data(img_data)

        ul_point, x_offset, y_offset, scale_factor = (10, 10), 50, 0, 0.5
        adj_keypoints = _adjust_keypoints(original_keypoints,
                                               ul_point,
                                               x_offset,
                                               y_offset,
                                               scale_factor)
        non_null_adj_keypoints = [point for point in adj_keypoints if point!=(0, 0)]

        ground_truth_path = join(self.datadir, 'adjusted_keypoints.pkl')
        with open(ground_truth_path, 'rb') as in_f:
            ground_truth = pickle.load(in_f)

        self.assertEqual(non_null_adj_keypoints, ground_truth)

    def test_get_kp_loss_mask(self):
        """
        Joints that are not found should have their losses
        masked during training.
        """
        from pose_detector.data_modules.dataset import _get_kp_loss_mask, _extract_keypoints_from_img_data
        img_data = self.dataset.imgs_data[0]
        keypoints = _extract_keypoints_from_img_data(img_data)

        kp_loss_mask = _get_kp_loss_mask(keypoints)
        desired_kp_loss_mask = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0]

        self.assertEqual(kp_loss_mask.tolist(), desired_kp_loss_mask)


class Image_Processing_Unittests(unittest.TestCase):
    """
    Unittest functions that are used to process raw image
    and image json data.
    """
    def setUp(self):
        self.datadir = join(UNITTESTS_DIRECTORY, 'data/pose_detector_data')

        json_path = join(self.datadir, 'test.json')
        with open(json_path, 'r') as in_f:
            self.imgs_data = json.load(in_f)['annotations']

        self.test_img_path = join(self.datadir, 'test.png')

    def test_get_valid_imgs(self):
        """
        Test ability to return only images that
        contain at least one keypoint.
        """
        from pose_detector.data_modules.dataset import _get_valid_imgs

        valid_img_indicies = _get_valid_imgs(self.imgs_data)

        self.assertEqual([0], valid_img_indicies)

    def test_keypoint_extraction(self):
        """
        Test ability to parse the json to get correct
        keypoints. Should always be 18 joints, regardless
        of whether they have been found or not.
        """
        from pose_detector.data_modules.dataset import _extract_keypoints_from_img_data

        test_img_data = self.imgs_data[0]
        keypoints = _extract_keypoints_from_img_data(test_img_data)

        num_non_null_keypoints = sum([point!=(0, 0) for point in keypoints])

        # Account for the aritificially added neck joint.
        orig_num_keypoints = test_img_data['num_keypoints']
        if keypoints[1] != (0, 0):
            orig_num_keypoints += 1

        self.assertEqual(num_non_null_keypoints, orig_num_keypoints)
        self.assertEqual(len(keypoints), 18)

    def test_get_cropping_rectangle_from_img_data(self):
        """
        Test ability to produce information that cv2 uses for
        cropping from images bounding box info.
        """
        from pose_detector.data_modules.dataset import _get_cropping_rectangle_from_img_data

        test_img_data = self.imgs_data[0]

        cropping_rectangle = _get_cropping_rectangle_from_img_data(test_img_data)
        desired = ((143, 22), 155, 400)

        self.assertEqual(cropping_rectangle, desired)

    def test_resize_img(self):
        """
        Test resizing imgs so max dim==256
        """
        from pose_detector.data_modules.dataset import _resize_img

        img = cv2.imread(self.test_img_path)
        max_dim = 256

        resized_img, _ = _resize_img(img, max_dim=max_dim)
        self.assertEqual(resized_img.shape, (171, 256, 3))

    def test_pad_img(self):
        """
        Should be able to pad an image along smallest dim
        such that it forms a square.
        """
        from pose_detector.data_modules.dataset import _pad_img

        unpadded_img_path = join(self.datadir, 'unpadded_img.png')
        unpadded_img = cv2.imread(unpadded_img_path)

        x_offset = 50
        y_offset = 0
        padded_img = _pad_img(unpadded_img, x_offset, y_offset)

        ground_truth_img_path = join(self.datadir, 'padded_img.png')
        ground_truth_img = cv2.imread(ground_truth_img_path)

        self.assertTrue(np.array_equal(padded_img, ground_truth_img))

    def test_create_heat_map(self):
        """
        Test creation of heat map from a keypoint.
        """
        from pose_detector.data_modules.dataset import _create_heat_map

        keypoint = (100, 50)
        edge_size = 256
        sigma = 20

        heat_map = _create_heat_map(keypoint, edge_size, sigma)

        expected_path = join(self.datadir, 'test_heatmap.png')
        with open(expected_path, 'rb') as in_f:
            expected = pickle.load(in_f)

        # Can't do full comparision as precision errors arise.
        heat_map = np.around(heat_map, decimals=3)
        expected = np.around(expected, decimals=3)

        self.assertEqual(heat_map.tolist(), expected.tolist())

    def test_draw_part_affinity_field(self):
        """
        Test drawing of part affinity field from keypoint.
        """
        from pose_detector.data_modules.dataset import _draw_part_affinity_field
        start_point = [10, 20]
        end_point = [30, 40]
        canvas = np.zeros([256, 256, 2])
        p_a_f = _draw_part_affinity_field(start_point, end_point, canvas)

        expected_path = join(self.datadir, 'test_paf.png')
        with open(expected_path, 'rb') as in_f:
            expected_p_a_f = pickle.load(in_f)

        # Can't do full comparision as precision errors arise.
        p_a_f = np.around(p_a_f, decimals=3)
        expected_p_a_f = np.around(expected_p_a_f, decimals=3)
        self.assertEqual(p_a_f.tolist(), expected_p_a_f.tolist())


if __name__ == "__main__":
    unittest.main()
