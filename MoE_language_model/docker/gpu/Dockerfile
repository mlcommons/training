# Copyright 2024 Google LLC
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

FROM nvcr.io/nvidia/nemo:24.09
WORKDIR /app

RUN pip install git+https://github.com/NVIDIA/dllogger#egg=dllogger
RUN pip install datasets==2.20.0 hydra-core sentencepiece
RUN pip install "git+https://github.com/mlperf/logging.git"

# clone current repo
ARG GIT_REF=main
RUN git clone https://github.com/ZhiyuLi-goog/MoE_study.git && \
    cd MoE_study && \
    echo GIT_REF=${GIT_REF} && \
    git checkout ${GIT_REF}

RUN patch --directory=/opt/megatron-lm -p1 < MoE_study/clm/docker/gpu/megatron_core.patch
