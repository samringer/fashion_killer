import re
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import json


def get_images_from_outfit_page(base_url):

    with open('mock_header.json', 'r') as in_f:
        headers = json.load(in_f)

    request = Request(url=base_url, headers=headers)
    html = urlopen(request).read()
    soup = BeautifulSoup(html, "html.parser")

    imgs = soup.find_all('img')
    img_list = [i['src'] for i in imgs]

    for img_name in img_list:
        if is_valid_source(img_name):
            img_name = fix_thumbnail_url(img_name)
            print(img_name)


def fix_thumbnail_url(src):
    src = src.replace("$S$", "$XXL$")
    src = src.replace("=40&", "=513&")
    return src


def is_valid_source(src):
    small_img_regex = r"\$S\$"
    if re.search(small_img_regex, src):
        return 'products' in src

if __name__ == "__main__":
    tester_url = "https://www.asos.com/good-for-nothing/good-for-nothing-oversized-hoodie-in-neon-yellow-with-chest-logo/prd/11733061?clr=yellow&SearchQuery=&cid=13522&gridcolumn=3&gridrow=2&gridsize=4&pge=1&pgesize=72&totalstyles=141"
    tester_url = "https://www.asos.com/asos-design/asos-design-neon-midi-rib-bodycon-dress-with-open-back/prd/11045115?clr=neon-pink&SearchQuery=&cid=20243&gridcolumn=1&gridrow=1&gridsize=4&pge=1&pgesize=72&totalstyles=137"
    get_images_from_outfit_page(tester_url)
