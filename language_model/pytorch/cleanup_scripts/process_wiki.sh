#!/bin/bash
# Copyright 2018 MLBenchmark Group. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# invocation script to cleanup the wiki dataset
# Usage: ./process_wiki.sh <the wiki_?? files>
# example: ./process_wiki.sh 'sample_data/wiki_??'
# The resulted files will be placed in ./results

inputs=$1

pip install nltk

# Remove doc tag and title
python ./cleanup_file.py --data=$inputs --output_suffix='.1'

# Further clean up files
for f in ${inputs}; do
  ./clean.sh ${f}.1 ${f}.2
done

# Sentence segmentation
python ./do_sentence_segmentation.py --data=$inputs --input_suffix='.2' --output_suffix='.3'

mkdir -p ./results

## Choose file size method or number of packages by uncommenting only one of the following do_gather options
# Gather into fixed size packages
python ./do_gather.py --data=$inputs --input_suffix='.3' --block_size=26.92 --out_dir='./results'
