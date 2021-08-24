#FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
FROM tensorflow/tensorflow:1.15.2-gpu

RUN apt-get update
RUN apt-get update && apt-get install -y --no-install-recommends time \
    ca-certificates \
    build-essential \
    git \
    bzip2 

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

COPY . /workspace
WORKDIR /workspace
ENTRYPOINT ["python", "mlcube.py"]