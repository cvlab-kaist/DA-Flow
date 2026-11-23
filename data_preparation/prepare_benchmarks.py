"""Convert Sintel / Spring / TartanAir into the evaluation layout.

Output layout (``<output_root>``)::

    gt_videos/<video_id>.mp4            # high-quality frames encoded as video (input of degrade_videos.py)
    gt_flows/<video_id>/flow_000.npy    # (H, W, 2) forward flow frame k -> k+1
    manifest.csv

Datasets:
    sintel     MPI-Sintel training set (23 scenes)
    spring     Spring training set, left camera, forward flow (37 scenes); the 2x
               super-resolved GT flow is subsampled to the image resolution
    tartanair  TartanAir ``soulcity`` and ``westerndesert`` (Easy + Hard), left camera,
               following the validation setup of FlowSeek

Example:
    python data_preparation/prepare_benchmarks.py sintel --root /path/to/MPI-Sintel --output_root data/eval/Sintel
"""
import argparse
import csv
import glob
import os
import re
import shutil
import subprocess
import tempfile

import cv2
import numpy as np


def natural_key(path):
    m = re.findall(r"\d+", os.path.basename(path))
    return (int(m[-1]) if m else 10 ** 12, os.path.basename(path))


def read_flo(path):
    with open(path, "rb") as f:
        assert np.fromfile(f, np.float32, count=1)[0] == 202021.25, f"Invalid .flo file: {path}"
        w, h = np.fromfile(f, np.int32, count=2)
        return np.fromfile(f, np.float32, count=2 * w * h).reshape(h, w, 2)


def read_flo5(path):
    import h5py

    with h5py.File(path, "r") as f:
        return f["flow"][()]


def write_video(frame_paths, out_path, fps):
    """Encode frames with libx264 (yuv420p); falls back to OpenCV mp4v if ffmpeg is unavailable."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if shutil.which("ffmpeg"):
        with tempfile.TemporaryDirectory() as tmp:
            for i, p in enumerate(frame_paths):
                os.symlink(os.path.abspath(p), os.path.join(tmp, f"{i:06d}.png"))
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "%06d.png"),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", out_path], check=True)
        return
    first = cv2.imread(frame_paths[0])
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (first.shape[1], first.shape[0]))
    for p in frame_paths:
        writer.write(cv2.imread(p))
    writer.release()


def collect_sintel(root):
    specs = []
    for scene in sorted(os.listdir(os.path.join(root, "training", "final"))):
        frames = sorted(glob.glob(os.path.join(root, "training", "final", scene, "frame_*.png")), key=natural_key)
        flows = sorted(glob.glob(os.path.join(root, "training", "flow", scene, "frame_*.flo")), key=natural_key)
        if len(frames) >= 2 and len(flows) == len(frames) - 1:
            specs.append((scene, frames, flows, read_flo))
    return specs


def collect_spring(root):
    specs = []
    for scene in sorted(os.listdir(os.path.join(root, "train"))):
        frames = sorted(glob.glob(os.path.join(root, "train", scene, "frame_left", "frame_left_*.png")), key=natural_key)
        flows = sorted(glob.glob(os.path.join(root, "train", scene, "flow_FW_left", "flow_FW_left_*.flo5")), key=natural_key)
        if len(frames) >= 2 and len(flows) == len(frames) - 1:
            specs.append((scene, frames, flows, lambda p: read_flo5(p)[::2, ::2]))
    return specs


def collect_tartanair(root, environments):
    specs = []
    for env in environments:
        for level in ("Easy", "Hard"):
            level_dir = os.path.join(root, env, level)
            if not os.path.isdir(level_dir):
                continue
            for traj in sorted(os.listdir(level_dir)):
                frames = sorted(glob.glob(os.path.join(level_dir, traj, "image_left", "*_left.png")), key=natural_key)
                flows = sorted(glob.glob(os.path.join(level_dir, traj, "flow", "*_flow.npy")),
                               key=lambda p: int(os.path.basename(p).split("_")[0]))
                if len(frames) >= 2 and len(flows) == len(frames) - 1:
                    specs.append((f"{env}_{level.lower()}_{traj}", frames, flows, np.load))
    return specs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=["sintel", "spring", "tartanair"])
    p.add_argument("--root", required=True, help="Original dataset root.")
    p.add_argument("--output_root", required=True)
    p.add_argument("--tartanair_envs", nargs="+", default=["soulcity", "westerndesert"])
    p.add_argument("--fps", type=int, default=None, help="Video fps (default: Sintel 5, Spring 30, TartanAir 30).")
    args = p.parse_args()

    if args.dataset == "sintel":
        specs, fps = collect_sintel(args.root), 5
    elif args.dataset == "spring":
        specs, fps = collect_spring(args.root), 30
    else:
        specs, fps = collect_tartanair(args.root, args.tartanair_envs), 30
    fps = args.fps or fps
    if not specs:
        raise RuntimeError(f"No sequences found under {args.root}")

    rows = []
    for video_id, frames, flows, reader in specs:
        print(f"{video_id}: {len(frames)} frames")
        write_video(frames, os.path.join(args.output_root, "gt_videos", f"{video_id}.mp4"), fps)
        flow_dir = os.path.join(args.output_root, "gt_flows", video_id)
        os.makedirs(flow_dir, exist_ok=True)
        for i, fp in enumerate(flows):
            np.save(os.path.join(flow_dir, f"flow_{i:03d}.npy"), np.asarray(reader(fp), dtype=np.float32))
        rows.append({"video_id": video_id, "num_frames": len(frames), "num_flows": len(flows)})

    with open(os.path.join(args.output_root, "manifest.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "num_frames", "num_flows"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} sequences to {args.output_root}")


if __name__ == "__main__":
    main()
