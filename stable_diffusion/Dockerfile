ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:22.12-py3
FROM ${FROM_IMAGE_NAME}

ENV DEBIAN_FRONTEND=noninteractive

# apt dependencies
RUN apt-get update
RUN apt-get install -y ffmpeg libsm6 libxext6

# pip dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install pytorch_lightning==1.9.0
RUN pip uninstall opencv-python==4.7.0.72 -y
RUN rm -rf /usr/local/lib/python3.8/dist-packages/cv2/
RUN pip install opencv-python==4.8.0.74
RUN pip install  httpx==0.24.1

# install LDM
ADD . /diffusion
RUN chmod +x /diffusion/*.sh
WORKDIR /diffusion
