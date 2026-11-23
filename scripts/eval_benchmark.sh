#!/bin/bash
# Reproduce Tab. 1: run DA-Flow on a degraded benchmark and evaluate.
# Usage: bash scripts/eval_benchmark.sh Sintel   (Sintel | Spring | TartanAir)
set -e
NAME=${1:-Sintel}
DATA=data/eval/${NAME}
python inference.py --input ${DATA}/lq_videos/general --output results/${NAME} --daflow_path checkpoints/DA-Flow
python evaluate.py --pred_dir results/${NAME} --gt_dir ${DATA}/gt_flows --output results/${NAME}/metrics.json
