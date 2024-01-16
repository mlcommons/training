ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:22.12-py3
FROM ${FROM_IMAGE_NAME}

ENV DEBIAN_FRONTEND=noninteractive

# apt dependencies
RUN apt-get update
RUN apt-get install -y ffmpeg libsm6 libxext6

# pip dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# install LDM
ADD . /diffusion
RUN chmod +x /diffusion/*.sh
WORKDIR /diffusion
