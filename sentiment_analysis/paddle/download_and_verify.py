import paddle
import os
import paddle.dataset.common

URL = 'http://ai.stanford.edu/%7Eamaas/data/sentiment/aclImdb_v1.tar.gz'
MD5 = '7c2ac02c03563afcf9b574c7e56c153a'
DIR = os.path.expanduser('~/.cache/paddle/dataset/imdb')
PATH = os.path.join(DIR, URL.split('/')[-1])

# Download the dataset
paddle.dataset.common.download(URL, 'imdb', MD5)

# Verify MD5 checksum
if paddle.dataset.common.md5file(PATH) == MD5:
    print("\nPASSED!")
else:
    print("\nFAILED")
