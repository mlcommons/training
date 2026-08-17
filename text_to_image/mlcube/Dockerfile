FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

RUN apt-get update && apt-get install -y --no-install-recommends \
    git vim curl unzip \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /workspace/flux

COPY . .

RUN pip install --no-cache-dir \
    huggingface-hub \ 
    -r torchtitan/requirements.txt \
    -r torchtitan/torchtitan/experiments/flux/requirements-flux.txt \
    -r torchtitan/requirements-mlperf.txt

RUN pip install -e torchtitan/

