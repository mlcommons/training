FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04

# Set working directory
WORKDIR /mlperf

RUN apt-get update
RUN apt-get install -y git make build-essential libssl-dev zlib1g-dev libbz2-dev \
                       libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
                       xz-utils tk-dev cmake unzip

# pyenv Install
RUN git clone https://github.com/pyenv/pyenv.git .pyenv

ENV HOME /mlperf
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

# Install Anaconda
RUN PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install anaconda3-5.0.1
RUN pyenv rehash
RUN pyenv global anaconda3-5.0.1

# Install PyTorch Requirements
ENV CMAKE_PREFIX_PATH "$(dirname $(which conda))/../"
RUN conda install -y numpy pyyaml mkl mkl-include setuptools cmake cffi typing
RUN conda install -c pytorch -y magma-cuda90

# Install PyTorch
RUN mkdir github
WORKDIR /mlperf/github
RUN git clone --recursive https://github.com/pytorch/pytorch
WORKDIR /mlperf/github/pytorch
RUN git checkout v0.4.0
RUN git submodule update --init
RUN python setup.py clean
RUN python setup.py install

# Install ncf-pytorch
WORKDIR /mlperf/ncf
# TODO: Change to clone github repo
ADD . /mlperf/ncf
RUN pip install -r requirements.txt
WORKDIR /mlperf/experiment
ENTRYPOINT ["/mlperf/ncf/run_and_time.sh"]
