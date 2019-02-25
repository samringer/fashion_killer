import re
import json
import urllib
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup


def get_outfits_from_page(page_url):

    with open('mock_header.json', 'r') as in_f:
        headers = json.load(in_f)

    outfit_urls = set()

    try:
        request = Request(url=page_url, headers=headers)
        html = urlopen(request).read()
    except urllib.error.HTTPError:
        print('An HTTP error!')
        return set()
    else:
        soup = BeautifulSoup(html, "html.parser")
        hrefs = soup.find_all(href=True)

        for i in hrefs:
            href = i['href']
            if outfit_url_is_valid(href):
                outfit_urls.add(href)
        print(outfit_urls)
    return outfit_urls

def outfit_url_is_valid(url):
    return "prd" in url

if __name__ == "__main__":
    tester_url = "https://www.asos.com/women/a-to-z-of-brands/a-star-is-born/cat/?cid=21073"
    print(get_outfits_from_page(tester_url))
