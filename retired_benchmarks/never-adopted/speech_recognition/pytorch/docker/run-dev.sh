#!/bin/bash
nvidia-docker run \
  -v /mnt/disk/mnt_dir:/mnt/disk/mnt_dir:rw \
  -v /etc/passwd:/etc/passwd:ro \
  -it --rm --user $(id -u) ds2-cuda9cudnn7:gpu
