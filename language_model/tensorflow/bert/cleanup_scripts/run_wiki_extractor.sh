#!/bin/bash

git clone https://github.com/attardi/wikiextractor.git

cd wikiextractor

git checkout 3162bb6c3c9ebd2d15be507aa11d6fa818a454ac

# Back to <bert>/cleanup_scripts
cd .. 

# Run `WikiExtractor.py` to extract data from XML.
data_dir=${DATA_DIR:-./wiki}
python3 wikiextractor/WikiExtractor.py $data_dir/enwiki-20200101-pages-articles-multistream.xml -o $data_dir/text