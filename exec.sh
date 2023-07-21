#!/bin/sh
docker run -it --gpus all --shm-size 64G \
    -v /path/to/this/repository:/workspace/ \
    pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime bash
