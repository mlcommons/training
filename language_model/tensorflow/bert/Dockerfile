#FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
FROM ubuntu:18.04

RUN apt-get update
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    bzip2 \
    python3 \
    python3-pip \
    python3-setuptools

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY requirements.txt /requirements.txt
RUN pip3 install --no-cache-dir -r /requirements.txt

COPY . /workspace
WORKDIR /workspace
ENTRYPOINT ["python3", "mlcube.py"]