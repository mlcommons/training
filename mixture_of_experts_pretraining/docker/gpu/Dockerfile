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

FROM nvcr.io/nvidia/pytorch:24.09-py3
WORKDIR /app

ARG NEMO_REVISION=7f3da35e1
RUN git clone https://github.com/NVIDIA/NeMo.git && \
    rm -rf /opt/NeMo && \
    cd NeMo && \
    git checkout ${NEMO_REVISION} && \
    sed -i "/mamba-ssm/d" requirements/requirements_nlp.txt && \
    sed -i 's/tensorstore<0.1.46/tensorstore/g' requirements/requirements_nlp.txt && \
    sed -i "/triton>=3.1.0/d" requirements/requirements.txt && \
    pip install --no-build-isolation -e ".[nlp]"

# ## Megatron-core
ARG MCORE_REVISION=a616d459039ae103257f6a20922261ac11ccbdf6
RUN pip uninstall -y megatron-core && \
     rm -rf /opt/megatron-lm && \
     git clone https://github.com/NVIDIA/Megatron-LM.git && \
     cd Megatron-LM && \
     git checkout ${MCORE_REVISION} && \
     #sed -i 's/tensorstore==0.1.45/tensorstore/g' requirements/pytorch:24.07/requirements.txt && \
     echo MCORE_COMMIT_HASH=$(git rev-parse HEAD) && \
     pip install . && \
     cd megatron/core/datasets && \
     make
ENV PYTHONPATH "${PYTHONPATH}:/workspace/Megatron-LM"

RUN pip install git+https://github.com/NVIDIA/dllogger#egg=dllogger
RUN pip install datasets==2.20.0 hydra-core sentencepiece
RUN pip install "git+https://github.com/mlperf/logging.git"

# TODO change to mlcommon git
RUN git clone  -b lizhiyu/moe --filter=blob:none --sparse https://github.com/ZhiyuLi-goog/training.git training

WORKDIR /app/training

RUN git sparse-checkout set mixture_of_experts_pretraining
RUN patch --directory=/app/Megatron-LM -p1 < mixture_of_experts_pretraining/docker/gpu/megatron_core.patch

