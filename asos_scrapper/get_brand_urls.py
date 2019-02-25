import re
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import json


def get_brand_urls(base_url):
    """
    Args:
        base_url (str): The url of the page containing all the brands.
    """

    with open('mock_header.json', 'r') as in_f:
        headers = json.load(in_f)

    request = Request(url=base_url, headers=headers)
    html = urlopen(request).read()
    soup = BeautifulSoup(html, "html.parser")
    hrefs = soup.find_all(href=True)

    brand_urls = set()
    for i in hrefs:
        href = i['href']
        if _brand_url_is_valid(href):
            brand_urls.add(href)

    return brand_urls


def _brand_url_is_valid(url):
    bad_endings = ['shop+by+brand',
                   'top+brands',
                   '(right)',
                   'new+edits',
                   'featured+brands',
                   '?cid=1340']
    brand_str = r"www\.asos\.com/women/a-to-z-of-brands/"
    if re.search(brand_str, url):
        for bad_ending in bad_endings:
            if url.endswith(bad_ending):
                return False
        return True

if __name__ == "__main__":
    tester_url = "http://www.asos.com/women/a-to-z-of-brands/cat/?cid=1340"
    print(get_brand_urls(tester_url))
