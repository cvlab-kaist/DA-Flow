"""DA-Flow flow estimation network.

A RAFT-style estimator whose matching features combine
(i) query/key features of the lifted image restoration diffusion transformer,
    upsampled with DPT heads, and
(ii) RAFT CNN features.
The context branch additionally receives DPT-upsampled diffusion features.
"""
import copy
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .corr import CorrBlock
from .custom_dpt import CustomDPTHead
from .extractor import BasicEncoder
from .update import BasicUpdateBlock
from .utils.utils import coords_grid

autocast = torch.cuda.amp.autocast

DIT_DIM = 1536  # hidden size of SD3.5-medium / DiT4SR


class DAFlow(nn.Module):
    def __init__(self, mixed_precision=False, dpt_channels=256, dropout=0.0):
        super().__init__()
        self.mixed_precision = mixed_precision
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        self.corr_levels = 4
        self.corr_radius = 4

        # RAFT CNN encoders (initialized from RAFT-things)
        self.fnet = BasicEncoder(output_dim=256, norm_fn="instance", dropout=dropout)
        self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn="batch", dropout=dropout)
        update_args = SimpleNamespace(corr_levels=self.corr_levels, corr_radius=self.corr_radius)
        self.update_block = BasicUpdateBlock(update_args, hidden_dim=hdim, use_context_dpt=True)

        # DPT heads that upsample diffusion features to 1/8 resolution
        self.upsampler_query = CustomDPTHead(dim_in=DIT_DIM, features=dpt_channels, feature_only=True)
        self.upsampler_key = copy.deepcopy(self.upsampler_query)
        self.dit_context_dpt = CustomDPTHead(dim_in=DIT_DIM, features=128, feature_only=True)

    @staticmethod
    def initialize_flow(img):
        N, _, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8, device=img.device)
        coords1 = coords_grid(N, H // 8, W // 8, device=img.device)
        return coords0, coords1

    @staticmethod
    def upsample_flow(flow, mask):
        """Upsample flow [H/8, W/8, 2] -> [H, W, 2] with convex combination."""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)
        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    @staticmethod
    def _to_bchw(feats, H, W):
        """Diffusion tokens -> (B, C, H/16, W/16).

        Accepts (B, T, C) token sequences or (B, h, w, C) feature maps.
        """
        out = []
        for f in feats:
            if f.dim() == 3:
                f = f.reshape(f.shape[0], H // 16, W // 16, f.shape[-1])
            out.append(f.permute(0, 3, 1, 2))
        return out

    def forward(self, image1, image2, features, iters=12, flow_init=None, test_mode=False):
        """
        Args:
            image1, image2: LQ frames in [-1, 1], (B, 3, H, W).
            features: [query_features, key_features], each a list (one entry per
                diffusion layer) of tokens of frame 1 (query) and frame 2 (key).
        """
        H, W = image1.shape[-2:]
        image1 = image1.contiguous()
        image2 = image2.contiguous()
        hdim, cdim = self.hidden_dim, self.context_dim

        feat1 = self._to_bchw(features[0], H, W)
        feat2 = self._to_bchw(features[1], H, W)

        # Diffusion matching features + context features
        dit_context = self.dit_context_dpt(feat1, image1)
        fmap1 = self.upsampler_query(feat1, image1)
        fmap2 = self.upsampler_key(feat2, image2)

        # Complementary CNN features
        with autocast(enabled=self.mixed_precision):
            raft_fmap1, raft_fmap2 = self.fnet([image1, image2])
        fmap1 = torch.cat([fmap1, raft_fmap1.float()], dim=1)
        fmap2 = torch.cat([fmap2, raft_fmap2.float()], dim=1)

        corr_fn = CorrBlock(fmap1, fmap2, num_levels=self.corr_levels, radius=self.corr_radius)

        with autocast(enabled=self.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)
        inp = torch.cat([dit_context, inp], dim=1)

        coords0, coords1 = self.initialize_flow(image1)
        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for _ in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)
            flow = coords1 - coords0
            with autocast(enabled=self.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)
            coords1 = coords1 + delta_flow
            flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
        return flow_predictions
