FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04


WORKDIR /research

RUN apt-get update

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    git \
    python \
    python-pip


ENV HOME /research
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH


RUN apt-get install -y python-setuptools

RUN apt-get install -y python-pip python3-pip virtualenv htop
RUN pip3 install --upgrade numpy scipy sklearn tf-nightly-gpu


# Mount data into the docker
ADD . /research/resnet


WORKDIR /research/resnet
RUN pip3 install -r official/requirements.txt


ENTRYPOINT ["/bin/bash"]

