# MI350X path — implements docs/training_recipe.md §"MI350X".

FROM rocm/primus:v26.3

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace/recommendation_v4

# torch / torchvision / torchaudio — training_recipe.md:38-40.
RUN pip install --upgrade --no-deps \
        --index-url https://download.pytorch.org/whl/rocm7.2 \
        torch==2.12.0+rocm7.2 \
        torchvision==0.27.0+rocm7.2 \
        torchaudio==2.11.0+rocm7.2

# torchrec — training_recipe.md:43.
RUN pip install --force-reinstall --no-deps \
        "git+https://github.com/pytorch/torchrec.git@v2026.06.01.00"

# fbgemm_gpu — training_recipe.md:42. Build from FBGEMM commit 10b77573 for
# gfx950 against the replaced torch. ~30-60 min.
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential && \
    rm -rf /var/lib/apt/lists/* && \
    git clone --recursive https://github.com/pytorch/FBGEMM.git /tmp/FBGEMM && \
    cd /tmp/FBGEMM && \
    git checkout 10b775730212923f65f7b78f79b6a01d80cf3c29 && \
    git submodule update --init --recursive && \
    cd fbgemm_gpu && \
    # Filter `fairscale` and the torch family from fbgemm's requirements.txt:
    # fairscale pulls a CPU torch that would clobber the +rocm7.2 wheel installed
    # above. fairscale is a distributed-training lib used by fbgemm tests, not
    # by the build itself.
    grep -v -E '^(fairscale|torch|torchvision|torchaudio)([<>=!]|$)' requirements.txt > /tmp/req.txt && \
    pip install -r /tmp/req.txt && \
    python setup.py -j 32 bdist_wheel \
        --build-target=default \
        --build-variant=rocm \
        -DHIP_ROOT_DIR=/opt/rocm \
        -DAMDGPU_TARGETS=gfx950 && \
    pip install --force-reinstall --no-deps dist/fbgemm_gpu_nightly_rocm*.whl && \
    cd / && rm -rf /tmp/FBGEMM

# polars-u64-idx — training_recipe.md:44 (mandatory; yambda-5b > 4.29 B rows).
# Remaining packages — training_recipe.md:156-159 ("Additional Python deps") plus
# `datasets` + `huggingface_hub`, which the recipe does not list but
# preprocess_public_data.py:278 imports to download yambda from HuggingFace.
RUN pip install \
        polars-u64-idx==1.33.1 \
        gin-config \
        absl-py \
        datasets \
        huggingface_hub \
        pyre-extensions \
        iopath \
        typing-inspect \
        psutil \
        tqdm \
        pyyaml \
        lightning-utilities && \
    # torchmetrics and tensordict declare `torch` as a dep; without --no-deps
    # pip pulls torch==2.12.0+cu130 from PyPI which clobbers the +rocm7.2 wheel
    # we installed above (libtorch_hip.so disappears, fbgemm_gpu fails to load).
    pip install --no-deps \
        torchmetrics==1.0.3 \
        tensordict

# mlperf_logging — required by train/mlperf_logging_utils.py for MLPerf
# compliance logs. Pinned to the Training 6.0 tag for reproducibility; --no-deps
# so pip does not resolve requirements.txt's torch/fbgemm_gpu/torchrec pins and
# clobber the +rocm7.2 wheels above.
RUN pip install --no-deps "git+https://github.com/mlcommons/logging.git@6.0.0-rc6"

# Smoke-test the 6 imports the launch script checks at
# scripts/launch_smoke_8gpu.sh:26.
RUN python -c "import torch, fbgemm_gpu, torchrec, polars, xxhash, gin; \
print('torch', torch.__version__, '| hip', getattr(torch.version, 'hip', None))"

COPY . /workspace/recommendation_v4

ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HSTU_HAMMER_KERNEL=TRITON \
    DLRM_DATA_PATH=/data/mlperf_dlrm_v4

CMD ["bash"]
