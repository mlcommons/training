FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel

WORKDIR /workspace/repository

RUN pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip install scikit-learn==0.24.2
RUN pip install torch_geometric==2.4.0
RUN pip install --no-index  torch_scatter==2.1.1 torch_sparse==0.6.17 -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
RUN pip install graphlearn-torch==0.2.2

RUN apt update
RUN apt install -y git
RUN pip install git+https://github.com/mlcommons/logging.git

# TF32 instead of FP32 for faster compute
ENV NVIDIA_TF32_OVERRIDE=1

RUN git clone https://github.com/alibaba/graphlearn-for-pytorch.git
WORKDIR /workspace/repository/graphlearn-for-pytorch/examples/igbh
