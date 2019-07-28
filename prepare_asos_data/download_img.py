import urllib.request


def download_img(img_url, output_path):
    urllib.request.urlretrieve(img_url, output_path)


if __name__ == "__main__":
    tester_url = "https://images.asos-media.com/products/asos-design-neon-midi-rib-bodycon-dress-with-open-back/11045115-4?$XXL$&wid=513&fit=constrain"
    tester_output_path = 'tester.png'
    download_img(tester_url, tester_output_path)

