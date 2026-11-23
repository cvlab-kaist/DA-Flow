import glob
import os

import cv2
import numpy as np
import torch

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def read_frames(path):
    """Read a video file or a directory of images as a list of RGB uint8 arrays."""
    if os.path.isdir(path):
        files = sorted(f for f in glob.glob(os.path.join(path, "*")) if f.lower().endswith(IMG_EXTS))
        return [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in files]
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def list_inputs(path):
    """Return (name, path) of the videos to process.

    ``path`` can be a video file, a directory of frames, or a directory containing
    video files and/or frame directories.
    """
    if os.path.isfile(path):
        return [(os.path.splitext(os.path.basename(path))[0], path)]
    entries = sorted(os.listdir(path))
    if any(e.lower().endswith(IMG_EXTS) for e in entries):
        return [(os.path.basename(os.path.normpath(path)), path)]
    inputs = []
    for e in entries:
        p = os.path.join(path, e)
        if os.path.isdir(p) or e.lower().endswith(VIDEO_EXTS):
            inputs.append((os.path.splitext(e)[0], p))
    return inputs


def frames_to_tensor(frames, multiple=16):
    """uint8 RGB frames -> (F, 3, H', W') tensor in [-1, 1], reflect-padded to a multiple of ``multiple``."""
    x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
    h, w = x.shape[-2:]
    ph, pw = (-h) % multiple, (-w) % multiple
    if ph or pw:
        x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
    return x * 2.0 - 1.0


def load_flow(path):
    """Load a flow map as (H, W, 2) float32 from .npy / .flo / KITTI-style .png."""
    if path.endswith(".npy"):
        flow = np.load(path)
        if flow.ndim == 3 and flow.shape[0] == 2:
            flow = flow.transpose(1, 2, 0)
        return flow.astype(np.float32)
    if path.endswith(".flo"):
        with open(path, "rb") as f:
            magic = np.fromfile(f, np.float32, count=1)
            assert magic == 202021.25, f"Invalid .flo file: {path}"
            w, h = np.fromfile(f, np.int32, count=2)
            data = np.fromfile(f, np.float32, count=2 * w * h)
        return data.reshape(h, w, 2)
    if path.endswith(".png"):
        flow = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32)
        return (flow[:, :, :2] - 2 ** 15) / 64.0
    raise ValueError(f"Unsupported flow format: {path}")


def write_flo(path, flow):
    """Write an (H, W, 2) flow to Middlebury .flo."""
    h, w = flow.shape[:2]
    with open(path, "wb") as f:
        np.array([202021.25], np.float32).tofile(f)
        np.array([w, h], np.int32).tofile(f)
        flow.astype(np.float32).tofile(f)


def write_flow_png(path, flow):
    """Write an (H, W, 2) flow as a 16-bit KITTI-style png (valid channel = 1)."""
    encoded = np.clip(np.round(flow * 64.0 + 2 ** 15), 0, 65535).astype(np.uint16)
    valid = np.ones(flow.shape[:2] + (1,), np.uint16)
    cv2.imwrite(path, np.concatenate([encoded, valid], axis=-1)[..., ::-1])
