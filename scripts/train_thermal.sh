#!/usr/bin/env bash

# GMFlow without refinement

# number of gpus for training, please set according to your hardware
# by default use all gpus on a machine
# can be trained on 4x 16GB V100 or 2x 32GB V100 or 2x 40GB A100 gpus
NUM_GPUS=2

# things (our final model is trained for 800K iterations, for ablation study, you can train for 200K)
CHECKPOINT_DIR=checkpoints/things-gmflow && \
mkdir -p ${CHECKPOINT_DIR} && \
python3 main.py \
--checkpoint_dir ${CHECKPOINT_DIR} \
--stage thermal \
--batch_size 8 \
--val_dataset thermal \
--lr 2e-4 \
--feature_channels 96 \
--image_size 384 768 \
--padding_factor 16 \
--upsample_factor 8 \
--with_speed_metric \
--val_freq 40000 \
--save_ckpt_freq 50000 \
--num_steps 200000 \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log
