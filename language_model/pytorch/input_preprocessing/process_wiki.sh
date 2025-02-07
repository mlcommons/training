#!/bin/bash

# invocation script to cleanup the wiki dataset
# Usage: ./process_wiki.sh <the wiki_?? files>
# example: ./process_wiki.sh 'sample_data/wiki_??'
# The resulted files will be placed in ./results

inputs=$1

pip install nltk

# Remove doc tag and title
# python ./cleanup_file.py --data=$inputs --output_suffix='.1'

# Further clean up files
# for f in ${inputs}; do
#   ./clean.sh ${f}.1 ${f}.2
# done

# Sentence segmentation
# python ./do_sentence_segmentation.py --data=$inputs --input_suffix='.2' --output_suffix='.3'

mkdir -p ./results

# Train/Eval seperation
python ./seperate_test_set.py --data=$inputs --input_suffix='.3' --output_suffix='.4' --num_test_articles=10000 --test_output='./results/eval'

## Choose file size method or number of packages by uncommenting only one of the following do_gather options
# Gather into fixed size packages
python ./do_gather.py --data=$inputs --input_suffix='.4' --block_size=26.92 --out_dir='./results'

# Gather into fixed number of packages
#NUM_PACKAGES=512
#python ./do_gather.py --data=$inputs --input_suffix='.3' --num_outputs=$NUM_PACKAGES --out_dir='./results'
