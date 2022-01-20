ARG FROM_IMAGE_NAME=pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
FROM ${FROM_IMAGE_NAME}

# Set working directory
WORKDIR /workspace/ssd

# Install requirements
COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt

COPY . .
