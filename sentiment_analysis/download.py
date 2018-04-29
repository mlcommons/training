from __future__ import print_function

import os
import requests
import shutil
import sys

URL = 'http://ai.stanford.edu/%7Eamaas/data/sentiment/aclImdb_v1.tar.gz'
DIR = os.path.expanduser('~/.cache/paddle/dataset/imdb')
PATH = os.path.join(DIR, URL.split('/')[-1])
CHUNK_SIZE = 4096

# Download the dataset
def download():
    if not os.path.exists(DIR):
        os.makedirs(DIR)

    retry = 0
    retry_limit = 3 
    while not os.path.exists(PATH):
        if retry < retry_limit:
            retry += 1
        else:
            raise RuntimeError("Cannot download {0} within retry limit {1}".
                               format(URL, retry_limit))
        r = requests.get(URL, stream=True)
        total_length = r.headers.get('content-length')

        if total_length is None:
            with open(PATH, 'w') as f:
                shutil.copyfileobj(r.raw, f)
        else:
            with open(PATH, 'w') as f:
                for data in r.iter_content(chunk_size=CHUNK_SIZE):
                    f.write(data)

    print("Download successful!")

if __name__ == "__main__":
    download()
