FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-devel

ADD . /

WORKDIR /

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-utils \
                       libglib2.0-0 \
                       libsm6 \
                       libxext6 \
                       libxrender-dev \
                       bc && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

RUN git clone https://github.com/cocodataset/cocoapi.git && \
    cd cocoapi/PythonAPI && \
    git reset --hard ed842bffd41f6ff38707c4f0968d2cfd91088688 && \
    python setup.py build_ext install

RUN python setup.py clean build develop --user
