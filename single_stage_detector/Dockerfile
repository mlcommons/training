# Copyright (c) 2018-2021, NVIDIA CORPORATION. All rights reserved.
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

ARG FROM_IMAGE_NAME=pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
# ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:22.01-py3
FROM ${FROM_IMAGE_NAME}

# Make sure git is installed (needed by pip)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        git \
 && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt /
# Cython needs to be present before attempting to install pycocotools
RUN pip install --no-cache-dir Cython
RUN pip install --no-cache-dir -r /requirements.txt

# Copy code
COPY . /workspace/single_stage_detector

# Set working directory
WORKDIR /workspace/single_stage_detector/ssd
