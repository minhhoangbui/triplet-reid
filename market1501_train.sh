#!/bin/sh
#
# This file calls train.py with all hyperparameters as for the TriNet
# experiment on market1501 in the original paper.

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 PATH_TO_IMAGES RESNET_CHECKPOINT_FILE EXPERIMENT_ROOT ..."
    echo "See the README for more info"
    echo "Download ResNet-50 checkpoint from https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models"
    exit 1
fi

# Shift the arguments so that we can just forward the remainder.
IMAGE_ROOT=$1 ; shift
INIT_CHECKPT=$1 ; shift
EXP_ROOT=$1 ; shift

python train.py \
    --train_set /mnt/hdd3tb/Users/phai/data/Market-1501-v15.09.15/market1501_train.csv \
    --model_name mobilenet_v1_1_224 \
    --image_root $IMAGE_ROOT \
    --initial_checkpoint $INIT_CHECKPT \
    --experiment_root $EXP_ROOT \
    --flip_augment \
    --crop_augment \
    --embedding_dim 128 \
    --head_name fc1024_normalize \
    --batch_p 18 \
    --batch_k 4 \
    --pre_crop_height 288 --pre_crop_width 144 \
    --net_input_height 256 --net_input_width 128 \
    --margin 0.6 \
    --loss batch_combined \
    --learning_rate 3e-4 \
    --train_iterations 25000 \
    --decay_start_iteration 15000 \
    "$@"
