FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04
WORKDIR /research
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3.6 \
    python3.6-dev \
    python3-pip \
    python3-setuptools
ENV HOME /research
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
# use numpy 1.16.4 for better compatibility with tensorflow-gpu 1.14.0
RUN pip3 install wheel && pip3 install \
    numpy==1.16.4 \
    scipy \
    sklearn \
    tensorflow-gpu==1.14.0

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Mount data into the docker
ADD . /research/transformer
WORKDIR /research/transformer
ENTRYPOINT ["/bin/bash"]
