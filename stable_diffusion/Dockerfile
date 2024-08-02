ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:24.01-py3
FROM ${FROM_IMAGE_NAME}

ENV DEBIAN_FRONTEND=noninteractive
ENV RCLONE_VER=v1.67.0

# apt dependencies
RUN apt-get update
RUN apt-get install -y ffmpeg libsm6 libxext6

# Install rclone from upstream, see https://github.com/mlcommons/training/issues/751
RUN wget https://github.com/rclone/rclone/releases/download/${RCLONE_VER}/rclone-${RCLONE_VER}-linux-amd64.zip \
    && unzip rclone-${RCLONE_VER}-linux-amd64.zip \
    && cd rclone-${RCLONE_VER}-linux-amd64 \
    && cp rclone /usr/bin/ \
    && chmod 755 /usr/bin/rclone \
    && rm -rf /rclone-${RCLONE_VER}-linux-amd64* \
    && rclone --version

# Remove the opencv version shipped with the base image
# https://github.com/opencv/opencv-python/issues/884
RUN pip uninstall -y opencv
RUN rm -rf /usr/local/lib/python3.10/dist-packages/cv2/

# install LDM
COPY . /diffusion
RUN cd /diffusion && \
    pip install --no-cache-dir -r requirements.txt
