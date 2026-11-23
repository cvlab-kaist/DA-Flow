#!/bin/bash
# Download pretrained weights.
#   bash scripts/download_weights.sh          # inference
#   bash scripts/download_weights.sh --train  # inference + training
# Stable Diffusion 3.5 Medium is gated: accept its license on the model page and run `huggingface-cli login` first.
set -e
DAFLOW_REPO=${DAFLOW_REPO:-Min-Jaewon/DA-Flow}
SD3=preset/models/stable-diffusion-3.5-medium

# DA-Flow (lifted transformer + flow network)
huggingface-cli download ${DAFLOW_REPO} --local-dir checkpoints/DA-Flow

# SD3.5 Medium: VAE and scheduler
huggingface-cli download stabilityai/stable-diffusion-3.5-medium --local-dir ${SD3} \
    --include "model_index.json" "vae/*" "scheduler/*"

if [ "$1" == "--train" ]; then
    # SD3.5 Medium text encoders (stage-1 prompts)
    huggingface-cli download stabilityai/stable-diffusion-3.5-medium --local-dir ${SD3} \
        --include "tokenizer*/*" "text_encoder/config.json" "text_encoder/model.safetensors" \
        "text_encoder_2/config.json" "text_encoder_2/model.safetensors" \
        "text_encoder_3/config.json" "text_encoder_3/model-0000*-of-00002.safetensors" "text_encoder_3/model.safetensors.index.json"

    # DiT4SR-Q (stage-1 initialization)
    huggingface-cli download acceptee/DiT4SR --local-dir preset/models --include "dit4sr_q/*"

    # SEA-RAFT (pseudo ground-truth flow in stage 2)
    huggingface-cli download MemorySlices/Tartan-C-T-TSKH-spring540x960-M --local-dir preset/models/sea_raft --include "model.safetensors"

    # RAFT C+T (stage-2 initialization)
    wget -O /tmp/raft_models.zip https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip
    unzip -j -o /tmp/raft_models.zip "models/raft-things.pth" -d preset/models
    rm /tmp/raft_models.zip
fi
