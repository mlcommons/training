# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

ARG NEMO_BASE_IMAGE=nvcr.io/nvidia/nemo:24.12-rc0
FROM ${NEMO_BASE_IMAGE} AS nemo-base-image

RUN pip uninstall transformers -y
RUN pip install transformers==4.47.1 blobfile==3.0.0
RUN pip install prettytable==3.12.0
RUN pip install git+https://github.com/mlcommons/logging.git@4.1.0-rc3

# setup workspace
WORKDIR /workspace/llama31
COPY . .

# Fixes the validation dataset order
RUN patch --directory=/opt/megatron-lm -p1 < mcore.patch
