import pickle
import urllib.request
from shutil import rmtree
from os import mkdir, system
from os.path import join

from get_images_from_outfit_page import get_images_from_outfit_page

with open('outfit_urls.pkl', 'rb') as in_f:
    outfit_urls = pickle.load(in_f)

outfit_urls = list(outfit_urls)

with open('outfit_urls_list.pkl', 'rb') as in_f:
    outfit_urls = pickle.load(in_f)

for dir_num, outfit_url in enumerate(outfit_urls):
    if dir_num < 56220:
        print(outfit_url)
        continue
    dir_name = str(dir_num).zfill(7)
    mkdir(dir_name)

    image_urls = get_images_from_outfit_page(outfit_url)

    for img_num, img_url in enumerate(image_urls):
        output_path = join(dir_name, str(img_num))
        try:
            urllib.request.urlretrieve(img_url, output_path)
        except urllib.error.HTTPError:
            print('An HTTP error!')

    # Do the scp in a hacky way.
    target_dir = "sam@samsergei.hopto.org:/home/sam/data/asos/train/"
    system('scp -r {dir_} {dest}'.format(dir_=dir_name, dest=target_dir))

    # Remove dir locally so we don't clog up system.
    rmtree(dir_name)
