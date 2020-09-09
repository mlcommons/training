FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime

# Set working directory
WORKDIR /mlperf/ssd

RUN apt-get update && \
    apt-get install -y python3-tk python-pip && \
    apt-get install -y numactl

RUN pip install --upgrade pip

# Necessary pip packages
COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir cython \
 && pip install --no-cache-dir https://github.com/mlperf/logging/archive/9ea0afa.zip \
 && pip install --no-cache-dir -r /requirements.txt

# Copy SSD code
COPY ssd .
