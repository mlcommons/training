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

ARG FROM_IMAGE_NAME=pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
FROM ${FROM_IMAGE_NAME}

ENV PYTORCH_VERSION=1.7.0a0+7036e91 

RUN apt-get update && \
    apt-get install -y libsndfile1 sox git cmake jq && \
    apt-get install -y --no-install-recommends numactl && \
    rm -rf /var/lib/apt/lists/*

RUN COMMIT_SHA=f546575109111c455354861a0567c8aa794208a2 && \
    git clone https://github.com/HawkAaron/warp-transducer deps/warp-transducer && \
    cd deps/warp-transducer && \
    git checkout $COMMIT_SHA && \
    sed -i 's/set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30 -O2")/#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30 -O2")/g' CMakeLists.txt && \
    sed -i 's/set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75")/set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_80,code=sm_80")/g' CMakeLists.txt && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make VERBOSE=1 && \
    export CUDA_HOME="/usr/local/cuda" && \
    export WARP_RNNT_PATH=`pwd` && \
    export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME && \
    export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH" && \
    export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH && \
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH && \
    export CFLAGS="-I$CUDA_HOME/include $CFLAGS" && \
    cd ../pytorch_binding && \
    python3 setup.py install && \
    rm -rf ../tests test ../tensorflow_binding && \
    cd ../../..

WORKDIR /workspace/rnnt

RUN pip install --no-cache --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110==0.28.0

RUN pip install --global-option="--cpp_ext" --global-option="--cuda_ext" https://github.com/NVIDIA/apex/archive/8a1ed9e8d35dfad26fb973996319965e4224dcdd.zip

COPY requirements.txt .
RUN pip install --no-cache --disable-pip-version-check -U -r requirements.txt

COPY . .
