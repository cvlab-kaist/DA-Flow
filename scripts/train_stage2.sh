#!/bin/bash
# Stage 2: train the DA-Flow network on the frozen lifted transformer.
# Paper setting: 4 GPUs x batch 4 x grad. accum. 2 = 32, lr 5e-5, 20K steps.
set -e
NUM_GPUS=${NUM_GPUS:-4}
STAGE1_CKPT=${STAGE1_CKPT:-experiments/stage1_lifting/checkpoint-20000}
accelerate launch --num_processes ${NUM_GPUS} --mixed_precision fp16 train_stage2_flow.py \
    --pretrained_model_name_or_path preset/models/stable-diffusion-3.5-medium \
    --transformer_model_name_or_path ${STAGE1_CKPT} \
    --raft_model_path preset/models/raft-things.pth \
    --sea_raft_model_path preset/models/sea_raft/model.safetensors \
    --train_root data/YouHQ/YouHQ-Train-frames \
    --output_dir experiments/stage2_flow \
    --n_frames 3 \
    --feature_layers 3 13 16 17 \
    --train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --max_train_steps 20000 \
    --checkpointing_steps 5000 \
    --mixed_precision fp16 \
    "$@"
