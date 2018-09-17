FROM pytorch/pytorch:0.4_cuda9_cudnn7

# Set working directory
WORKDIR /mlperf

RUN apt-get update && \
    apt-get install -y python3-tk python-pip

# Necessary pip packages
RUN pip install --upgrade pip
RUN pip install Cython==0.28.4 \
                matplotlib==2.2.2
RUN python3 -m pip install pycocotools==2.0.0

# Copy SSD code
WORKDIR /mlperf
COPY . .
RUN pip install -r requirements.txt

WORKDIR /mlperf/ssd
