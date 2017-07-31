#!/bin/bash
DATASET_DIR=/mnt/lustre/DATAshare/imagenet_tensorflow
TRAIN_DIR=./train_log



export LD_LIBRARY_PATH=/mnt/lustre/share/cuda-8.0-cudnn5.1/lib64/:$LD_LIBRARY_PATH



if [ ! -d "$TRAIN_DIR" ]; then
  mkdir $TRAIN_DIR
fi
now=$(date +"%Y%m%d_%H%M%S")   # for log
srun -p TITANXP --job-name=kk_v3_test --gres=gpu:8 python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=imagenet \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=kk_v2 \
    2>&1|tee $TRAIN_DIR/kk_test_v2-$now.log &\
