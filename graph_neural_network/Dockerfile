FROM nvcr.io/nvidia/pytorch:22.12-py3

WORKDIR /workspace/repository

RUN pip install scikit-learn==0.24.2
RUN pip install torch_geometric==2.4.0
RUN pip install torch_scatter==2.1.1 torch_sparse==0.6.17
RUN pip install graphlearn-torch==0.2.2

RUN apt update
RUN apt install -y git
RUN pip install git+https://github.com/mlcommons/logging.git

# TF32 instead of FP32 for faster compute
ENV NVIDIA_TF32_OVERRIDE=1

COPY . .
WORKDIR /workspace/repository

RUN git clone https://github.com/alibaba/graphlearn-for-pytorch.git
WORKDIR /workspace/repository/graphlearn-for-pytorch
RUN git checkout 910cb55
RUN git submodule update --init
