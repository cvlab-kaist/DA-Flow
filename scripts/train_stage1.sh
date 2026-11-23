#!/bin/bash
# Stage 1: lift DiT4SR to video (full spatio-temporal attention, F=3).
# Paper setting: 4 GPUs x batch 1 x grad. accum. 8 = 32, lr 5e-5, 20K steps.
set -e
NUM_GPUS=${NUM_GPUS:-4}
accelerate launch --num_processes ${NUM_GPUS} --mixed_precision fp16 train_stage1_lifting.py \
    --pretrained_model_name_or_path preset/models/stable-diffusion-3.5-medium \
    --transformer_model_name_or_path preset/models/dit4sr_q \
    --train_root data/YouHQ/YouHQ-Train-frames \
    --output_dir experiments/stage1_lifting \
    --n_frames 3 \
    --null_text_ratio 0.2 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --max_train_steps 20000 \
    --checkpointing_steps 5000 \
    --mixed_precision fp16 \
    "$@"
