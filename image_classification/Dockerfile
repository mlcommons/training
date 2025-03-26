FROM tensorflow/tensorflow:2.4.0-gpu

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && apt-get install -y --no-install-recommends git wget unzip

# Install python dependencies
COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt

# Copy code
COPY . /workspace
RUN chmod +x /workspace/*.sh

# Set working directory
WORKDIR /workspace