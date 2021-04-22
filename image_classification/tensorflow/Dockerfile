FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04


WORKDIR /research

RUN apt-get update

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    git \
    python \
    python-pip \
    python3-dev  

COPY ./nccl-repo-ubuntu1604-2.5.6-ga-cuda9.0_1-1_amd64.deb /
RUN apt-get install libnccl2=2.5.6-1+cuda9.0 libnccl-dev=2.5.6-1+cuda9.0


ENV HOME /research
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH


RUN apt-get install -y python-setuptools

RUN apt-get install -y python-pip python3-pip virtualenv htop
RUN pip3 install --upgrade numpy==1.16.4 scipy==0.16.1 scikit-learn==0.22.2.post1 futures==3.1.1 h5py==2.10.0 google-crc32c==1.0.0 tensorflow-gpu==1.10.0


# Mount data into the docker
ADD . /research/resnet


WORKDIR /research/resnet
RUN pip3 install -r official/requirements.txt



ENTRYPOINT ["/bin/bash"]
