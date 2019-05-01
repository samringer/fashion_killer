from pathlib import Path
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
        if len(list((outfit_dir).glob('*'))) != 4:
            print('{} does not contain 4 images'.format(outfit_dir))
            shutil.rmtree(outfit_dir.as_posix())
            continue
        for img_path in outfit_dir.iterdir():
            img = cv2.imread(img_path.as_posix())
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            inp = np.asarray(img) / 256
            pose_img = monkey.draw_pose_from_img(inp)
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


if __name__ == "__main__":
    dir_path = '/home/sam/data/asos/2604_clean'
    clean_asos_dir(dir_path)
