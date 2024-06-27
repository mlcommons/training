ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:24.04-py3
FROM ${FROM_IMAGE_NAME}

# Copy code
WORKDIR /workspace/llm
COPY . .
RUN pip install -r requirements.txt
ENV PYTHONPATH "/workspace/llm:${PYTHONPATH}"
