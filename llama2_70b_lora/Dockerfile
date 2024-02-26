ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:24.01-py3
FROM ${FROM_IMAGE_NAME}

WORKDIR /workspace/ft-llm
ADD . /workspace/ft-llm

RUN pip install -r requirements.txt
RUN pip install flash-attn==2.4.1 --no-build-isolation
