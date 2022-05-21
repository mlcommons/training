from __future__ import print_function

import hashlib
import os

URL = 'http://ai.stanford.edu/%7Eamaas/data/sentiment/aclImdb_v1.tar.gz'
MD5 = '7c2ac02c03563afcf9b574c7e56c153a'
DIR = os.path.expanduser('~/.cache/paddle/dataset/imdb')
PATH = os.path.join(DIR, URL.split('/')[-1])
CHUNK_SIZE = 4096

def md5content(fname):
    hash_md5 = hashlib.md5()
    f = open(fname, "rb")
    for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
        hash_md5.update(chunk)
    f.close()
    return hash_md5.hexdigest()

# Verify MD5 checksum
def verify():
    if md5content(PATH) == MD5:
        print("PASSED!")
    else:
        print("FAILED")

if __name__ == "__main__":
    verify()
