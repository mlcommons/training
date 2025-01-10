ARG FROM_IMAGE_NAME=pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
FROM ${FROM_IMAGE_NAME}

ENV DEBIAN_FRONTEND=noninteractive

# apt dependencies
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-get update && apt-get install -y
RUN apt-get install -y ffmpeg libsm6 libxext6 git wget unzip \
        build-essential \
        libomp-dev

# pip dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install pytorch_lightning==1.9.0
RUN pip uninstall opencv-python==4.7.0.72 -y
RUN rm -rf /usr/local/lib/python3.8/dist-packages/cv2/
RUN pip install opencv-python==4.8.0.74
RUN pip install numpy==1.26.4
RUN pip install  httpx==0.24.1
# RUN pip install --upgrade pip setuptools wheel
RUN pip install fastapi==0.115.6
RUN pip install starlette==0.45.2
RUN pip install jinja2==2.11.3
RUN pip install triton==3.1.0

# install LDM
ADD . /diffusion
RUN chmod +x /diffusion/*.sh
WORKDIR /diffusion