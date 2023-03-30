#FROM tensorflow/tensorflow:1.15.2-gpu
FROM tensorflow/tensorflow:2.4.0-gpu

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        git libcurl4 wget unzip \
 && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt

# Copy code
COPY . /workspace/bert

RUN chmod +x /workspace/bert/cleanup_scripts/*.sh

# Set working directory
WORKDIR /workspace/bert
