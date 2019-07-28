import pickle

from get_brand_urls import get_brand_urls
from get_all_next_pages import get_all_next_pages
from get_outfits_from_page import get_outfits_from_page

base_url = "http://www.asos.com/women/a-to-z-of-brands/cat/?cid=1340"
brand_urls = get_brand_urls(base_url)
with open('brand_urls.pkl', 'wb') as in_f:
    pickle.dump(brand_urls, in_f)

all_page_urls = set()
for brand_url in brand_urls:
    page_urls = get_all_next_pages(brand_url)
    all_page_urls = all_page_urls.union(page_urls)
with open('page_urls.pkl', 'wb') as in_f:
    pickle.dump(all_page_urls, in_f)

with open('page_urls.pkl', 'rb') as in_f:
    all_page_urls = pickle.load(in_f)

all_outfit_pages = set()
for page_url in all_page_urls:
    outfit_pages = get_outfits_from_page(page_url)
    all_outfit_pages = all_outfit_pages.union(outfit_pages)
with open('outfit_urls.pkl', 'wb') as in_f:
    pickle.dump(all_outfit_pages, in_f)
