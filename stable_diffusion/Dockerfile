ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:24.01-py3
FROM ${FROM_IMAGE_NAME}

ENV DEBIAN_FRONTEND=noninteractive

# apt dependencies
RUN apt-get update
RUN apt-get install -y ffmpeg libsm6 libxext6

# Remove the opencv version shipped with the base image
# https://github.com/opencv/opencv-python/issues/884
RUN pip uninstall -y opencv
RUN rm -rf /usr/local/lib/python3.10/dist-packages/cv2/

# install LDM
COPY . /diffusion
RUN cd /diffusion && \
    pip install --no-cache-dir -r requirements.txt
