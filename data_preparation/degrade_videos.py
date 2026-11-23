"""Synthesize degraded (LQ) videos from high-quality videos.

Each video is degraded with the second-order Real-ESRGAN pipeline at 1/4 scale plus
video compression, then bicubically upsampled x4 back to (about) the input size.
DA-Flow uses the default ``general`` setting for both training and evaluation.

Example:
    python data_preparation/degrade_videos.py --input_dir data/eval/Sintel/gt_videos \
        --output_dir data/eval/Sintel/lq_videos/general
"""
import argparse
import glob
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
from tqdm import tqdm

from degradation import degradation_process


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_video(path):
    """-> (1, 3, T, H, W) in [-1, 1], fps"""
    frames, _, info = torchvision.io.read_video(path, pts_unit="sec", output_format="TCHW")
    video = frames.float() / 255.0 * 2.0 - 1.0
    return rearrange(video, "t c h w -> 1 c t h w"), info.get("video_fps", 30)


@torch.no_grad()
def degrade(video, difficulty="general"):
    """(1, 3, T, H, W) in [-1, 1] -> degraded and x4-upsampled video in [-1, 1]."""
    lq, _ = degradation_process(video, difficulty=None if difficulty == "general" else difficulty)
    t = lq.shape[2]
    lq = F.interpolate(rearrange(lq, "b c t h w -> (b t) c h w"), scale_factor=4, mode="bicubic")
    return rearrange(lq, "(b t) c h w -> b c t h w", t=t)


def save_video(video, path, fps):
    """(3, T, H, W) in [-1, 1] -> h264 mp4"""
    x = ((video.clamp(-1, 1) + 1) / 2).mul(255).add_(0.5).clamp_(0, 255)
    x = x.permute(1, 2, 3, 0).to("cpu", torch.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torchvision.io.write_video(path, x, fps=int(fps), video_codec="h264")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True, help="Directory of high-quality .mp4 videos.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--difficulty", default="general", choices=["easy", "general", "medium", "hard"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    videos = sorted(glob.glob(os.path.join(args.input_dir, "*.mp4")))
    for path in tqdm(videos):
        out_path = os.path.join(args.output_dir, os.path.basename(path))
        if os.path.exists(out_path):
            continue
        video, fps = load_video(path)
        # Input is cast to bf16 as in the original data generation config
        lq = degrade(video.to(device, torch.bfloat16), args.difficulty)
        save_video(lq[0], out_path, fps)


if __name__ == "__main__":
    main()
