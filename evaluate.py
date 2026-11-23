"""Evaluate predicted optical flow against ground truth.

Reports EPE and the outlier rates (% of pixels with EPE > 1 / 3 / 5 px), averaged
over all frame pairs of all videos (pixels with |GT flow| >= 400 are ignored).
Predictions and GT are compared on their common top-left region.

Expected layout (``flow_XXX`` = flow from frame XXX to XXX+1):
    pred_dir/<video>/flow_000.{npy,flo,png}
    gt_dir/<video>/flow_000.{npy,flo,png}
"""
import argparse
import glob
import json
import os

import numpy as np

from daflow.utils.io import load_flow

MAX_FLOW = 400


def frame_metrics(pred, gt, max_flow=MAX_FLOW):
    # Compare on the common top-left region (the degradation pipeline crops frames
    # to a multiple of 16, e.g. Sintel 436 -> 432 rows).
    h, w = min(pred.shape[0], gt.shape[0]), min(pred.shape[1], gt.shape[1])
    pred, gt = pred[:h, :w], gt[:h, :w]
    epe = np.sqrt(np.sum((pred - gt) ** 2, axis=-1))
    valid = np.sqrt(np.sum(gt ** 2, axis=-1)) < max_flow
    if not valid.any():
        return None
    epe = epe[valid]
    return {"epe": float(epe.mean()), "1px": float((epe >= 1).mean()), "3px": float((epe >= 3).mean()), "5px": float((epe >= 5).mean())}


def find_flow(directory, idx):
    for ext in (".npy", ".flo", ".png"):
        path = os.path.join(directory, f"flow_{idx:03d}{ext}")
        if os.path.isfile(path):
            return path
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--gt_dir", required=True)
    p.add_argument("--output", default=None, help="Optional json file for per-video results.")
    args = p.parse_args()

    per_video, all_frames = {}, []
    for video in sorted(os.listdir(args.pred_dir)):
        pred_video = os.path.join(args.pred_dir, video)
        gt_video = os.path.join(args.gt_dir, video)
        if not os.path.isdir(pred_video):
            continue
        if not os.path.isdir(gt_video):
            print(f"[skip] no GT flow for {video}")
            continue
        frames = []
        num_gt = len(glob.glob(os.path.join(gt_video, "flow_*")))
        for idx in range(num_gt):
            pred_path, gt_path = find_flow(pred_video, idx), find_flow(gt_video, idx)
            if pred_path is None or gt_path is None:
                break
            m = frame_metrics(load_flow(pred_path), load_flow(gt_path))
            if m is not None:
                frames.append(m)
        if frames:
            per_video[video] = {k: float(np.mean([f[k] for f in frames])) for k in frames[0]}
            all_frames.extend(frames)

    if not all_frames:
        raise RuntimeError("No matching prediction / GT flow pairs were found.")
    summary = {k: float(np.mean([f[k] for f in all_frames])) for k in all_frames[0]}
    print(f"Videos: {len(per_video)}, frame pairs: {len(all_frames)}")
    print(f"EPE: {summary['epe']:.3f} | 1px: {100 * summary['1px']:.2f} | 3px: {100 * summary['3px']:.2f} | 5px: {100 * summary['5px']:.2f}  (outlier %)")
    if args.output:
        with open(args.output, "w") as f:
            json.dump({"summary": summary, "per_video": per_video}, f, indent=2)


if __name__ == "__main__":
    main()
