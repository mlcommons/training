docker pull nvcr.io/nvidia/pytorch:23.09-py3
docker run -v path_to_my_folder:/root/workspace --workdir /root/workspace --gpus all -it --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/pytorch:23.09-py3
