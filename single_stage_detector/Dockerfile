FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime

# Set working directory
WORKDIR /mlperf

RUN apt-get update && \
    apt-get install -y python3-tk python-pip && \
    apt-get install -y numactl

RUN pip install --upgrade pip

# Copy SSD code
WORKDIR /mlperf
COPY . .
# Necessary pip packages
RUN pip install -r requirements.txt
RUN python3 -m pip install pycocotools==2.0.0

WORKDIR /mlperf/ssd
