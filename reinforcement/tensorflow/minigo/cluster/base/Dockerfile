FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

# Set the locale to UTF-8. If this isn't set, the Python library `choix`
# fails to install for some reason.
ENV LC_ALL="C.UTF-8"
ENV LC_CTYPE="C.UTF-8"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-10-0 \
        cuda-cublas-dev-10-0 \
        cuda-cudart-dev-10-0 \
        cuda-cufft-dev-10-0 \
        cuda-curand-dev-10-0 \
        cuda-cusolver-dev-10-0 \
        cuda-cusparse-dev-10-0 \
        curl \
        git \
        libcurl3-dev \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        libstdc++6 \
        pkg-config \
        rsync \
        software-properties-common \
        unzip \
        zip \
        zlib1g-dev \
        gdb \
        python3 \
        python3-numpy \
        python3-dev \
        python3-pip \
        python3-wheel \
        python3-setuptools \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" && \
    echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update -y && apt-get install google-cloud-sdk -y

ENV BAZEL_VERSION 0.24.1

WORKDIR /
RUN mkdir /bazel && \
    cd /bazel && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

ENV CI_BUILD_PYTHON python3

run mv /usr/bin/python /usr/bin/python.bak
run ln -s /usr/bin/python3 /usr/bin/python
ENV CUDA_HOME /usr/local/cuda
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
ENV LD_LIBRARY_PATH ${CUDA_HOME}/lib64:${CUDA_HOME}/lib64/stubs/:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

WORKDIR /app

# NVIDIA's CUDA docker image has cuDNN installed in /usr not /usr/local/cuda.
ENV CUDNN_INSTALL_PATH /usr

# For the slow stuff, copy just the needed scripts so we don't break the docker cache
COPY staging/WORKSPACE WORKSPACE
COPY staging/.bazelrc .bazelrc
COPY staging/cc/tensorflow/BUILD cc/tensorflow/BUILD
COPY staging/cc/tensorflow/copy_outputs.sh cc/tensorflow/copy_outputs.sh
COPY staging/cc/configure_tensorflow.sh cc/configure_tensorflow.sh
ENV CC_OPT_FLAGS -march=ivybridge
ADD staging/requirements.txt /app/requirements.txt

RUN pip3 install --upgrade pip setuptools
#TODO: just install what we need (keras?) so changes to requirements.txt don't trigger TF rebuilds
RUN pip3 install -r /app/requirements.txt

ENV TF_NEED_TENSORRT 0
RUN cc/configure_tensorflow.sh
