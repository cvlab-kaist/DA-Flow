"""DA-Flow inference pipeline.

Given a clip of LQ frames, the lifted DiT4SR transformer is run along a 10-step
flow-matching schedule. At every denoising step, query/key features of the
selected layers are cached for each frame, and the DA-Flow network estimates
forward flow between consecutive frames from these features. By default the flow
of step index 1 (the second of ten steps) is returned, so only two denoising
passes are needed.
"""
import re
from typing import List, Optional

import numpy as np
import torch

DIT_DIM = 1536


def _gaussian_weights(tile_w, tile_h):
    """Gaussian blending weights for overlapping tiles, shape (tile_h, tile_w)."""
    var = 0.01
    mid = (tile_w - 1) / 2
    x_probs = [np.exp(-(x - mid) * (x - mid) / (tile_w * tile_w) / (2 * var)) / np.sqrt(2 * np.pi * var) for x in range(tile_w)]
    mid = tile_h / 2
    y_probs = [np.exp(-(y - mid) * (y - mid) / (tile_h * tile_h) / (2 * var)) / np.sqrt(2 * np.pi * var) for y in range(tile_h)]
    return torch.tensor(np.outer(y_probs, x_probs))


def _tile_grid(h, w, tile_size, tile_overlap):
    """Number of tiles along width (rows) and height (cols)."""
    grid_rows = 0
    cur_x = 0
    while cur_x < w:
        cur_x = max(grid_rows * tile_size - tile_overlap * grid_rows, 0) + tile_size
        grid_rows += 1
    grid_cols = 0
    cur_y = 0
    while cur_y < h:
        cur_y = max(grid_cols * tile_size - tile_overlap * grid_cols, 0) + tile_size
        grid_cols += 1
    return grid_rows, grid_cols


def _tile_offsets(h, w, tile_size, tile_overlap, grid_rows, grid_cols):
    """Top-left offsets (ofs_y, ofs_x) of overlapping square tiles covering an (h, w) map."""
    offsets = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            ofs_x = max(row * tile_size - tile_overlap * row, 0)
            ofs_y = max(col * tile_size - tile_overlap * col, 0)
            if row == grid_rows - 1:
                ofs_x = w - tile_size
            if col == grid_cols - 1:
                ofs_y = h - tile_size
            offsets.append((ofs_y, ofs_x))
    return offsets


class DAFlowPipeline:
    def __init__(
        self,
        vae,
        transformer,
        flow_net,
        scheduler,
        prompt_embeds,
        pooled_prompt_embeds,
        feature_layers=(3, 13, 16, 17),
        n_frames=3,
    ):
        self.vae = vae
        self.transformer = transformer
        self.flow_net = flow_net
        self.scheduler = scheduler
        self.prompt_embeds = prompt_embeds  # (1, L, 4096) null-prompt embedding
        self.pooled_prompt_embeds = pooled_prompt_embeds  # (1, 2048)
        self.feature_layers = list(feature_layers)
        self.n_frames = n_frames
        self.transformer.lifting_n_frames = n_frames

        # Hooks on the query / key projections of the selected blocks. Each projection is
        # called twice per forward (image stream, then LQ control stream); DA-Flow uses the
        # image-stream outputs, i.e. entries 4*l (query) and 4*l + 1 (key) for layer l.
        self._features: List[torch.Tensor] = []
        pattern = re.compile(r"transformer_blocks\.(\d+)\.attn\.(to_q|to_k)$")
        for name, module in self.transformer.named_modules():
            m = pattern.match(name)
            if m and int(m.group(1)) in self.feature_layers:
                module.register_forward_hook(lambda _m, _i, out: self._features.append(out))

    @property
    def device(self):
        return next(self.transformer.parameters()).device

    def _run_transformer(self, latents, control, timestep, prompt_embeds, pooled):
        self._features.clear()
        noise_pred = self.transformer(
            hidden_states=latents,
            controlnet_image=control,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled,
            return_dict=False,
        )[0]
        return noise_pred, list(self._features)

    def _denoise_clip(self, latents, control, timestep, latent_tiled_size, latent_tiled_overlap):
        """One transformer pass over a clip of n_frames frames (with latent tiling for large inputs).

        Returns the noise prediction (F, 16, h, w) and per-layer (query, key) features (F, h/2, w/2, C).
        """
        num = latents.shape[0]
        h, w = latents.shape[-2:]
        prompt_embeds = self.prompt_embeds.repeat(num, 1, 1)
        pooled = self.pooled_prompt_embeds
        timestep = timestep.expand(num)

        # The DiT4SR control branch assumes a square token grid, so non-square inputs are
        # always processed with square tiles.
        if h == w and h * w <= latent_tiled_size * latent_tiled_size:
            noise_pred, feats = self._run_transformer(latents, control, timestep, prompt_embeds, pooled)
            feats = [f.reshape(-1, h // 2, w // 2, f.shape[-1]) for f in feats]
            return noise_pred, feats

        tile_size = min(latent_tiled_size, min(h, w))
        feat_tile, feat_overlap = tile_size // 2, latent_tiled_overlap // 2
        tile_weights = _gaussian_weights(tile_size, tile_size).to(self.device)
        tile_weights = tile_weights[None, None].expand(1, 16, -1, -1)
        feat_weights = _gaussian_weights(feat_tile, feat_tile).to(self.device)
        feat_weights = feat_weights[None, :, :, None].expand(1, -1, -1, DIT_DIM)

        # The feature map (h/2, w/2) uses the same tile grid as the latent.
        grid = _tile_grid(h, w, tile_size, latent_tiled_overlap)
        offsets = _tile_offsets(h, w, tile_size, latent_tiled_overlap, *grid)
        feat_offsets = _tile_offsets(h // 2, w // 2, feat_tile, feat_overlap, *grid)

        noise_pred = torch.zeros(latents.shape, device=self.device)
        contributors = torch.zeros(latents.shape, device=self.device)
        stitched, feat_contrib = None, None
        for (oy, ox), (fy, fx) in zip(offsets, feat_offsets):
            tile_pred, tile_feats = self._run_transformer(
                latents[:, :, oy:oy + tile_size, ox:ox + tile_size],
                control[:, :, oy:oy + tile_size, ox:ox + tile_size],
                timestep, prompt_embeds, pooled,
            )
            noise_pred[:, :, oy:oy + tile_size, ox:ox + tile_size] += tile_pred * tile_weights
            contributors[:, :, oy:oy + tile_size, ox:ox + tile_size] += tile_weights

            if stitched is None:
                shape = (num, h // 2, w // 2, tile_feats[0].shape[-1])
                stitched = [torch.zeros(shape, device=self.device) for _ in tile_feats]
                feat_contrib = [torch.zeros(shape, device=self.device) for _ in tile_feats]
            for k, f in enumerate(tile_feats):
                f = f.reshape(-1, feat_tile, feat_tile, f.shape[-1])
                stitched[k][:, fy:fy + feat_tile, fx:fx + feat_tile, :] += f * feat_weights
                feat_contrib[k][:, fy:fy + feat_tile, fx:fx + feat_tile, :] += feat_weights

        noise_pred /= contributors
        feats = [s / c for s, c in zip(stitched, feat_contrib)]
        return noise_pred, feats

    @staticmethod
    def _clip_ranges(num_frames, chunk_size, overlap):
        stride = chunk_size - overlap
        ranges = []
        for start in range(0, num_frames, stride):
            end = min(start + chunk_size, num_frames)
            if end - start < chunk_size:
                end = num_frames
                start = max(0, end - chunk_size)
            ranges.append((start, end))
        return ranges

    @torch.no_grad()
    def __call__(
        self,
        frames: torch.Tensor,
        latents: torch.Tensor,
        num_inference_steps: int = 10,
        flow_steps: Optional[List[int]] = (1,),
        clip_overlap: int = 1,
        latent_tiled_size: int = 64,
        latent_tiled_overlap: int = 24,
        flow_batch_size: int = 2,
        flow_iters: int = 12,
    ):
        """
        Args:
            frames: LQ frames in [-1, 1], (F, 3, H, W) with H, W multiples of 16.
            latents: initial noise, (F, 16, H/8, W/8).
            flow_steps: denoising step indices whose flow is returned. Denoising stops
                after the last requested step. ``None`` returns every step.
            flow_batch_size: frame pairs per flow-network call. The context encoder uses
                BatchNorm with batch statistics, so keep 2 to reproduce the paper.
        Returns:
            dict {step_index: forward flow (F-1, 2, H, W)}.
        """
        device = self.device
        dtype = self.transformer.dtype
        num_frames = frames.shape[0]
        frames = frames.to(device=device)
        frames_vae = frames.to(dtype=dtype)

        # Encode LQ frames (DiT4SR control latents)
        control = []
        for f in range(num_frames):
            z = self.vae.encode(frames_vae[f:f + 1]).latent_dist.sample()
            control.append((z - self.vae.config.shift_factor) * self.vae.config.scaling_factor)
        control = torch.cat(control, dim=0)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        latents = latents.to(device=device, dtype=dtype)
        h, w = latents.shape[-2:]
        last_step = len(timesteps) - 1 if flow_steps is None else max(flow_steps)

        flows = {}
        for i, t in enumerate(timesteps):
            noise_cache = [None] * num_frames
            feat_cache = [[[None] * num_frames for _ in range(2)] for _ in self.feature_layers]

            # Process the video in overlapping clips of n_frames frames; predictions and
            # features of frames shared by two clips are averaged.
            for start, end in self._clip_ranges(num_frames, self.n_frames, clip_overlap):
                noise_pred, feats = self._denoise_clip(
                    latents[start:end].clone(), control[start:end].clone(), t,
                    latent_tiled_size, latent_tiled_overlap,
                )
                for f in range(end - start):
                    idx = start + f
                    pred = noise_pred[f:f + 1]
                    noise_cache[idx] = pred if noise_cache[idx] is None else (noise_cache[idx] + pred) / 2.0
                for l in range(len(self.feature_layers)):
                    for qk in range(2):  # 0: query, 1: key (image stream)
                        per_frame = feats[4 * l + qk].chunk(end - start, dim=0)
                        for f in range(end - start):
                            idx = start + f
                            cur = per_frame[f]
                            prev = feat_cache[l][qk][idx]
                            feat_cache[l][qk][idx] = cur if prev is None else (prev + cur) / 2.0

            noise_pred = torch.cat(noise_cache, dim=0)
            q_feats = [torch.stack(feat_cache[l][0], 0).reshape(-1, h // 2, w // 2, DIT_DIM) for l in range(len(self.feature_layers))]
            k_feats = [torch.stack(feat_cache[l][1], 0).reshape(-1, h // 2, w // 2, DIT_DIM) for l in range(len(self.feature_layers))]
            del noise_cache, feat_cache

            if flow_steps is None or i in flow_steps:
                flows[i] = self._estimate_flow(frames, q_feats, k_feats, flow_batch_size, flow_iters)
            del q_feats, k_feats
            torch.cuda.empty_cache()

            if i >= last_step:
                break
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        return flows

    def _estimate_flow(self, frames, q_feats, k_feats, batch_size, iters):
        """Forward flow between consecutive frames (frame f -> f+1)."""
        out = []
        num_pairs = frames.shape[0] - 1
        for f in range(0, num_pairs, batch_size):
            e = min(f + batch_size, num_pairs)
            q = [feat[f:e] for feat in q_feats]
            k = [feat[f + 1:e + 1] for feat in k_feats]
            _, flow = self.flow_net(frames[f:e], frames[f + 1:e + 1], features=[q, k], iters=iters, test_mode=True)
            out.append(flow.float().cpu())
        return torch.cat(out, dim=0)
