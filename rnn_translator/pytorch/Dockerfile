FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

WORKDIR /workspace/pytorch

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .
