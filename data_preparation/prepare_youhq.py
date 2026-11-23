"""Build the YouHQ training set used by both training stages.

For every YouHQ clip (``<root>/<category>/<video>/<clip>.mp4``), a random 512x512
crop (shared by all frames) is taken, the crop is degraded with ``degrade_videos.degrade``,
and HQ / LQ frames are written as::

    <output_root>/YouHQ-Train-frames/<category>/<video>/<clip>_frames/frame_00.jpg ...
    <output_root>/YouHQ-Train-LQ-frames/<category>/<video>/<clip>_frames/frame_00.jpg ...
    <output_root>/YouHQ-Train-prompts/<category>/<video>/<clip>.txt   (if --prompt_root is given)

Captions are only used in stage 1 (80% of the time; null prompt otherwise).

Example:
    python data_preparation/prepare_youhq.py --root /path/to/YouHQ-Train --output_root data/YouHQ
"""
import argparse
import glob
import os
import random
import shutil

import cv2
import torch
from tqdm import tqdm

from degrade_videos import degrade, load_video, set_seed


def save_frames(video, out_dir):
    """(3, T, H, W) in [-1, 1] -> frame_XX.jpg"""
    os.makedirs(out_dir, exist_ok=True)
    x = ((video.clamp(-1, 1) + 1) / 2).mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8)
    for t in range(x.shape[1]):
        frame = x[:, t].permute(1, 2, 0).numpy()
        cv2.imwrite(os.path.join(out_dir, f"frame_{t:02d}.jpg"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="YouHQ-Train root: <category>/<video>/<clip>.mp4")
    p.add_argument("--output_root", required=True)
    p.add_argument("--prompt_root", default=None, help="Optional captions: <category>/<video>/<clip>.txt")
    p.add_argument("--split", default="Train")
    p.add_argument("--crop_size", type=int, default=512)
    p.add_argument("--difficulty", default="general")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rank", type=int, default=0, help="Process every world_size-th clip starting at rank.")
    p.add_argument("--world_size", type=int, default=1)
    args = p.parse_args()

    set_seed(args.seed + args.rank)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hq_root = os.path.join(args.output_root, f"YouHQ-{args.split}-frames")
    lq_root = os.path.join(args.output_root, f"YouHQ-{args.split}-LQ-frames")
    prompt_out = os.path.join(args.output_root, f"YouHQ-{args.split}-prompts")

    clips = sorted(glob.glob(os.path.join(args.root, "*", "*", "*.mp4")))[args.rank::args.world_size]
    for path in tqdm(clips):
        rel = os.path.splitext(os.path.relpath(path, args.root))[0]  # <category>/<video>/<clip>
        if os.path.isdir(os.path.join(lq_root, rel + "_frames")):
            continue
        video, _ = load_video(path)
        h, w = video.shape[-2:]
        s = args.crop_size
        if h < s or w < s:
            continue
        y, x = random.randint(0, h - s), random.randint(0, w - s)
        video = video[..., y:y + s, x:x + s]
        lq = degrade(video.to(device, torch.bfloat16), args.difficulty)

        save_frames(video[0], os.path.join(hq_root, rel + "_frames"))
        save_frames(lq[0, :, :, :s, :s], os.path.join(lq_root, rel + "_frames"))
        if args.prompt_root is not None:
            src = os.path.join(args.prompt_root, rel + ".txt")
            if os.path.isfile(src):
                os.makedirs(os.path.dirname(os.path.join(prompt_out, rel + ".txt")), exist_ok=True)
                shutil.copy(src, os.path.join(prompt_out, rel + ".txt"))


if __name__ == "__main__":
    main()
