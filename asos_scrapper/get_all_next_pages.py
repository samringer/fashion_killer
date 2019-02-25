import re
import json
import urllib
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup


def get_all_next_pages(page_1_url):
    """
    Args:
        page_1_url (str): url of the first page containing clothes.
    """

    page_urls = {page_1_url}
    current_url = page_1_url
    print(current_url)
    current_page_num = 1
    while True:
        current_url = get_next_page(current_url, current_page_num)
        if not current_url:
            break
        current_page_num = get_trailing_numbers(current_url)
        page_urls.add(current_url)
        print(current_url)
    return page_urls


def get_next_page(url, current_page_num):

    with open('mock_header.json', 'r') as in_f:
        headers = json.load(in_f)

    try:
        request = Request(url=url, headers=headers)
        html = urlopen(request).read()
    except urllib.error.HTTPError:
        print('An HTTP error!')
        return

    else:
        soup = BeautifulSoup(html, "html.parser")
        hrefs = soup.find_all(href=True)

        for i in hrefs:
            href = i['href']
            if page_url_is_valid(href, current_page_num):
                next_page_url = href
                return next_page_url
    return

def get_trailing_numbers(url):
    m = re.search(r'\d+$', url)
    return int(m.group()) if m else None


def page_url_is_valid(url, current_page_num):
    if "page=" in url:
        if get_trailing_numbers(url) > current_page_num:
            return True


if __name__ == '__main__':
    tester_url = "https://www.asos.com/women/a-to-z-of-brands/asos-collection/cat/?cid=4877&page=1"
    get_all_next_pages(tester_url)
