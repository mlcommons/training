#!/bin/bash
# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

DATA_ROOT_DIR="${DATA_ROOT_DIR:-/datasets}"

python ./utils/convert_librispeech.py \
    --input_dir ${DATA_ROOT_DIR}/LibriSpeech/train-clean-100 \
    --dest_dir ${DATA_ROOT_DIR}/LibriSpeech/train-clean-100-wav \
    --output_json ${DATA_ROOT_DIR}/LibriSpeech/librispeech-train-clean-100-wav.json
python ./utils/convert_librispeech.py \
    --input_dir ${DATA_ROOT_DIR}/LibriSpeech/train-clean-360 \
    --dest_dir ${DATA_ROOT_DIR}/LibriSpeech/train-clean-360-wav \
    --output_json ${DATA_ROOT_DIR}/LibriSpeech/librispeech-train-clean-360-wav.json
python ./utils/convert_librispeech.py \
    --input_dir ${DATA_ROOT_DIR}/LibriSpeech/train-other-500 \
    --dest_dir ${DATA_ROOT_DIR}/LibriSpeech/train-other-500-wav \
    --output_json ${DATA_ROOT_DIR}/LibriSpeech/librispeech-train-other-500-wav.json


python ./utils/convert_librispeech.py \
    --input_dir ${DATA_ROOT_DIR}/LibriSpeech/dev-clean \
    --dest_dir ${DATA_ROOT_DIR}/LibriSpeech/dev-clean-wav \
    --output_json ${DATA_ROOT_DIR}/LibriSpeech/librispeech-dev-clean-wav.json
python ./utils/convert_librispeech.py \
    --input_dir ${DATA_ROOT_DIR}/LibriSpeech/dev-other \
    --dest_dir ${DATA_ROOT_DIR}/LibriSpeech/dev-other-wav \
    --output_json ${DATA_ROOT_DIR}/LibriSpeech/librispeech-dev-other-wav.json


python ./utils/convert_librispeech.py \
    --input_dir ${DATA_ROOT_DIR}/LibriSpeech/test-clean \
    --dest_dir ${DATA_ROOT_DIR}/LibriSpeech/test-clean-wav \
    --output_json ${DATA_ROOT_DIR}/LibriSpeech/librispeech-test-clean-wav.json
python ./utils/convert_librispeech.py \
    --input_dir ${DATA_ROOT_DIR}/LibriSpeech/test-other \
    --dest_dir ${DATA_ROOT_DIR}/LibriSpeech/test-other-wav \
    --output_json ${DATA_ROOT_DIR}/LibriSpeech/librispeech-test-other-wav.json

bash scripts/create_sentencepieces.sh
