"""Minimal SEA-RAFT (Wang et al., ECCV 2024; BSD-3-Clause, see LICENSE in this folder).

Only used in stage 2 to produce pseudo ground-truth flow from the HQ frames.
Source: https://github.com/princeton-vl/SEA-RAFT
"""
import json
import os
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .raft import RAFT

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "config", "spring-M.json")


class SeaRaftFlowExtractor(nn.Module):
    """Frozen SEA-RAFT pseudo-label generator (spring-M config, 20 iterations)."""

    def __init__(self, model_path, cfg_path=DEFAULT_CONFIG, iters=20):
        super().__init__()
        with open(cfg_path) as f:
            self.args = Namespace(**json.load(f))
        self.args.iters = iters
        self.model = RAFT(self.args)
        if model_path.endswith(".safetensors"):  # HuggingFace release (MemorySlices/...)
            from safetensors.torch import load_file
            state_dict = load_file(model_path)
        else:  # Google Drive release (.pth)
            state_dict = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(self, image1, image2):
        """image1/2: (B, 3, H, W) HQ frames in [-1, 1]. Returns forward flow (B, 2, H, W).

        Note: the [-1, 1] frames are scaled by 255 as they are (not mapped to [0, 255]),
        exactly as in the training run of the paper; we keep it for reproducibility.
        """
        image1 = image1.float() * 255.0
        image2 = image2.float() * 255.0
        s = self.args.scale
        img1 = F.interpolate(image1, scale_factor=2 ** s, mode="bilinear", align_corners=False)
        img2 = F.interpolate(image2, scale_factor=2 ** s, mode="bilinear", align_corners=False)
        flow = self.model(img1, img2, iters=self.args.iters, test_mode=True)["flow"][-1]
        return F.interpolate(flow, scale_factor=0.5 ** s, mode="bilinear", align_corners=False) * (0.5 ** s)
