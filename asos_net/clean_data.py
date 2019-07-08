from pathlib import Path
from copy import deepcopy
import os
import pickle
import shutil

import numpy as np
import cv2

from rtc_server.monkey import Monkey


def clean_asos_dir(data_dir):
    """
    Removes 'outfits' that do not contain exactly 4 images.
    Removes 'outfits' that are not recognized by the pose detector.
    Args:
    data_dir (str): Path to directory to clean
    """
    data_dir = Path(data_dir)
    monkey = Monkey()

    for outfit_dir in data_dir.iterdir():
        # Check there are exactly 4 images
        dir_path = outfit_dir.as_posix()
        if len(list((outfit_dir).glob('*'))) != 4:
            print('{} does not contain 4 images'.format(outfit_dir))
            shutil.rmtree(outfit_dir.as_posix())
            continue
        total_valid_keypoints = 0
        for img_path in outfit_dir.iterdir():
            img = cv2.imread(img_path.as_posix())
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            inp = np.asarray(img) / 256
            pose_img, keypoints = monkey.draw_pose_from_img(inp)
            num_valid_kp = sum([1 if i != (0, 0) else 0
                            for i in keypoints])
            total_valid_keypoints += num_valid_kp
            mirrored_keypoints = get_mirror_image_keypoints(keypoints)
            rev_pose_img = monkey.pose_drawer.draw_pose_from_keypoints(mirrored_keypoints)
            # Check that each image in an 'outfit' has a corresponding
            # pose
            if pose_img.tolist() == np.zeros([256, 256, 3]).tolist():
                print('No pose in {}'.format(outfit_dir))
                shutil.rmtree(outfit_dir.as_posix())
                break

            # Save the pose images
            pose_img *= 256
            pose_img = pose_img.astype('uint8')
            cv2.imwrite(str(img_path.with_suffix('.pose.jpg')), pose_img)

            rev_pose_img *= 256
            rev_pose_img = rev_pose_img.astype('uint8')
            cv2.imwrite(str(img_path.with_suffix('.revpose.jpg')),
                        rev_pose_img)
        if os.path.isdir(dir_path):
            if total_valid_keypoints < 30:
                print("num valid keypoints = {} for {}".format(
                    total_valid_keypoints, outfit_dir))
                shutil.rmtree(outfit_dir.as_posix())
            else:
                print("{} done".format(outfit_dir))


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
    # 201 drops out of the way asos images are cropped by monkey
    mirrored_keypoints = [(201-x, y) if (x, y) != (0, 0) else (x, y)
                          for x, y in mirrored_keypoints]
    return mirrored_keypoints

if __name__ == "__main__":
    dir_path = '/home/sam/data/asos/0107_clean/'
    clean_asos_dir(dir_path)
