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

# TODO change to mlcommon git
RUN git clone  -b lizhiyu/moe --filter=blob:none --sparse https://github.com/ZhiyuLi-goog/training.git training

WORKDIR /app/training

RUN git sparse-checkout set MoE_language_model
RUN patch --directory=/opt/megatron-lm -p1 < MoE_language_model/docker/gpu/megatron_core.patch