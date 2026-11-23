"""Paired HQ / degraded-LQ video clip dataset (YouHQ layout).

Expected layout (``--train_root`` points to the HQ frame root)::

    YouHQ-Train-frames/<category>/<video>/<clip>_frames/frame_00.jpg ...    # HQ frames
    YouHQ-Train-LQ-frames/<category>/<video>/<clip>_frames/frame_00.jpg ... # degraded frames
    YouHQ-Train-prompts/<category>/<video>/<clip>.txt                       # captions (optional)

See ``data_preparation/prepare_youhq.py``.
"""
import glob
import os
import random

import torch
from einops import rearrange
from PIL import Image
from torch.utils import data
from torchvision import transforms


class VideoPairedCaptionDataset(data.Dataset):
    def __init__(self, root_folder, lq_suffix="-LQ-frames", prompt_suffix="-prompts", null_text_ratio=0.2, n_frames=3):
        super().__init__()
        root_folder = os.path.abspath(root_folder)
        if not root_folder.endswith("-frames"):
            raise ValueError(f"HQ frame root should end with '-frames', got {root_folder}")
        self.gt_root = root_folder
        self.lq_root = root_folder[: -len("-frames")] + lq_suffix
        self.prompt_root = root_folder[: -len("-frames")] + prompt_suffix
        self.null_text_ratio = null_text_ratio
        self.n_frames = n_frames
        self.to_tensor = transforms.ToTensor()
        self.clips = self._list_clips(root_folder)

    @staticmethod
    def _list_clips(root):
        """Relative clip paths <category>/<video>/<clip> (without the '_frames' suffix)."""
        clips = []
        for category in sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))):
            category_path = os.path.join(root, category)
            for video in sorted(d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))):
                for clip in sorted(os.listdir(os.path.join(category_path, video))):
                    clips.append(os.path.join(category, video, clip.replace("_frames", "")))
        return clips

    def _load(self, root, rel_path, idx):
        img = Image.open(os.path.join(root, rel_path + "_frames", f"frame_{idx:02d}.jpg")).convert("RGB")
        return self.to_tensor(img) * 2.0 - 1.0  # [-1, 1]

    def __getitem__(self, index):
        rel_path = self.clips[index]
        num_frames = len(glob.glob(os.path.join(self.lq_root, rel_path + "_frames", "*.jpg")))
        start = random.randint(0, num_frames - self.n_frames)

        lq_image = torch.stack([self._load(self.lq_root, rel_path, start + f) for f in range(self.n_frames)])
        gt_image = torch.stack([self._load(self.gt_root, rel_path, start + f) for f in range(self.n_frames)])

        prompt_path = os.path.join(self.prompt_root, rel_path + ".txt")
        if random.random() < self.null_text_ratio or not os.path.isfile(prompt_path):
            prompt = ""
        else:
            with open(prompt_path) as f:
                prompt = f.read().strip()

        # Frame pair (i < j) used for flow supervision in stage 2
        flow1_idx, flow2_idx = sorted(random.sample(range(self.n_frames), 2))
        return {
            "video_id": rel_path,
            "lq_image": lq_image,  # (F, 3, H, W)
            "gt_image": gt_image,
            "frame_idx": start,
            "prompt": prompt,
            "flow1_idx": flow1_idx,
            "flow2_idx": flow2_idx,
        }

    def __len__(self):
        return len(self.clips)


def collate_fn(batch):
    """Stacks clips and folds frames into the batch axis: (B*F, 3, H, W)."""
    return {
        "video_id": [b["video_id"] for b in batch],
        "prompt": [b["prompt"] for b in batch],
        "lq_image": rearrange(torch.stack([b["lq_image"] for b in batch]), "b f c h w -> (b f) c h w"),
        "gt_image": rearrange(torch.stack([b["gt_image"] for b in batch]), "b f c h w -> (b f) c h w"),
        "frame_idx": [b["frame_idx"] for b in batch],
        "flow1_idx": [b["flow1_idx"] for b in batch],
        "flow2_idx": [b["flow2_idx"] for b in batch],
    }
