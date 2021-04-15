#!/bin/bash

CONFIG=$1
GPUS=2
PORT=${PORT:-29500}

DEBUG=False \
CUDA_VISIBLE_DEVICES=2,3 \
PYTHONPATH="$(dirname $0)/../../..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}