FROM rocm/pytorch:rocm6.4_ubuntu22.04_py3.10_pytorch_release_2.6.0

WORKDIR /workspace

RUN pip install pybind11
RUN pip install ninja
RUN pip install packaging
RUN /usr/bin/python3 -m pip install pyYAML

# Install library dependencies
WORKDIR /workspace/deps

# FlashAttention
RUN git clone https://github.com/ROCm/flash-attention/ flash_attention \
    # latest stable commit of ck_tile/fa3 branch
    && cd flash_attention && git checkout cace3592812640486b04196a209bb85d12267b4c \
    && git submodule update --init --recursive \
    && PYTORCH_ROCM_ARCH='gfx942' GPU_ARCHS="gfx942" MAX_JOBS=64 pip install --no-build-isolation -e .

ADD patches /workspace/deps/patches

# Megatron-core
RUN git clone --recursive https://github.com/ROCm/Megatron-LM.git megatron_lm
RUN pip uninstall -y megatron-core
# dev branch commit
RUN cd megatron_lm && git checkout megatron_190213a_mlperf   \
    && pip install -e . && cd megatron/core/datasets && make

ENV PYTHONPATH "${PYTHONPATH}:/workspace/deps/megatron_lm"

# mambe dependency required for NeMo
RUN git clone https://github.com/state-spaces/mamba.git mamba_ssm \
    && cd mamba_ssm \
    && git checkout v2.2.2 \
    && export HIP_ARCHITECTURES="gfx942" \
    && pip install --no-cache-dir --verbose .

# NeMo
RUN git clone https://github.com/NVIDIA/NeMo nemo \
    && cd nemo && git checkout v2.1.0
RUN cd /workspace/deps/nemo \
    && git apply /workspace/deps/patches/nemo_v2_1_0.patch \
    && pip install --no-build-isolation -e ".[nlp]"

# NeMo-Run
RUN pip install git+https://github.com/NVIDIA/NeMo-Run.git@v0.4.0

# Python deps
# Important this should be done after NeMo, otherwise the pinned transformers==4.40.2 version will be overwritten
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Transformer Engine
ARG TE_COMMIT=te_v1.9_mlperf_llama2
RUN git clone --recursive https://github.com/ROCm/TransformerEngine.git \
    # dev branch commit
    && cd TransformerEngine && git checkout $TE_COMMIT && git submodule update --init --recursive \
    # Workaround logging debug info to the console
    && sed -i 's/self.logger.info/self.logger.debug/g' /workspace/deps/TransformerEngine/transformer_engine/pytorch/attention.py \
    && sed -i 's/warnings.warn/if False: warnings.warn/g' /workspace/deps/TransformerEngine/transformer_engine/pytorch/attention.py \
    && sed -i '/.*\"window_size should be.*/d' /workspace/deps/TransformerEngine/transformer_engine/common/fused_attn_rocm/fused_attn.cpp \
    && NVTE_FUSED_ATTN_AOTRITON=0 NVTE_ROCM_ARCH='gfx942' NVTE_FRAMEWORK='pytorch' NVTE_USE_HIPBLASLT=1 MAX_JOBS=128 PYTORCH_ROCM_ARCH='gfx942' GPU_ARCHS='gfx942' pip install  -e .

# Install hipBLASLt (FP8 tuned gemms - second round)
 RUN git clone https://github.com/ROCm/hipBLASLt.git \
     && cd hipBLASLt && git checkout ebc770851dfb99a1bbb8ef2e5873c753f8011a47 \
     && sudo apt-get update \
     && apt install -y python3.10-venv \
     && ./install.sh -idc -a gfx942

# RPD
RUN sudo apt-get update && \
    apt --fix-broken install -y && \
    apt-get install -y\
    sqlite3 libsqlite3-dev \
    libfmt-dev

RUN git clone https://github.com/ROCmSoftwarePlatform/rocmProfileData \
    && cd rocmProfileData \
    && cd rocpd_python \
    && python3 setup.py bdist_wheel \
    && pip install dist/*.whl \
    && cd .. \
    && cd rpd_tracer \
    && python3 setup.py bdist_wheel \
    && pip3 install dist/*.whl \
    && cd .. \
    && make; make install

WORKDIR /workspace/code

# Copy the current state of the code inside the image
COPY . .