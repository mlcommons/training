FROM us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.10_tpuvm 

WORKDIR /app

RUN pip install datasets==3.2.0 accelerate==1.2.1 evaluate==0.4.3 scikit-learn==1.6.0

# Install jax, jaxlib, libtpu nightly
RUN pip install --pre -U jax -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html
RUN pip install --pre -U jaxlib -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html
RUN pip install libtpu-nightly -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -U --pre

# Add --no-deps to avoid version dependency conflicts between all above libraries like jax, pallas, torch or torch_xla
RUN pip install torch_xla[pallas] -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html --no-deps

# custom transformers with static mixtral implementation
# TODO: import changes in the current repo
# branch: lizhiyu/dpo_static_default
ARG TRANSFORMERS_REVISION=6172624929ce75c0f0ececa776d70415b9829c75
RUN git clone https://github.com/pytorch-tpu/transformers && \
    cd transformers && \
    echo TRANSFORMERS_REVISION=${TRANSFORMERS_REVISION} && \
    git checkout ${TRANSFORMERS_REVISION} && \
    echo TRANSFORMERS_REVISION=$(git rev-parse HEAD) && \
    pip install -e .

# Add the Google Cloud SDK package repository
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# Install the Google Cloud SDK
RUN apt-get update && apt-get install -y google-cloud-sdk vim

RUN pip install hydra-core==1.3.2
RUN pip install tensorboard==2.18.0 tensorboardX==2.6.2.2
RUN pip install sentencepiece==0.2.0

# checkpoint loading in gcs
RUN pip install gcsfs==2024.12.0

# mlperf log
RUN pip install git+https://github.com/mlperf/logging.git@eb9e1a39bc313d964e9c1955d76384a6f3a731d3

# import schedulers from nemo
RUN pip install nemo_toolkit==1.23.0 pytorch-lightning==2.5.0.post0 huggingface-hub==0.23.2

WORKDIR /app

# TODO change to mlcommon git
RUN git clone  -b lizhiyu/moe --filter=blob:none --sparse https://github.com/ZhiyuLi-goog/training.git training

WORKDIR /app/training

RUN git sparse-checkout set mixture_of_experts_pretraining