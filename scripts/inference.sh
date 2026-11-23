#!/bin/bash
# Estimate optical flow for degraded videos.
# Usage: bash scripts/inference.sh <video | frame dir | dir of videos> <output dir>
set -e
INPUT=$1
OUTPUT=${2:-results/demo}
python inference.py --input ${INPUT} --output ${OUTPUT} --daflow_path checkpoints/DA-Flow --save_vis
