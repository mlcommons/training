FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

WORKDIR /research

RUN apt-get update


ENV HOME /research
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential


RUN apt-get install -y python-pip python3-pip virtualenv htop
RUN pip3 install --upgrade numpy scipy sklearn tf-nightly-gpu
RUN pip install numpy scipy tf-nightly-gpu

WORKDIR /research/resnet

ADD . /research/resnet
RUN pip install -r requirements.txt

WORKDIR /research

ENTRYPOINT ["/bin/bash"]

