"""DA-Flow inference: estimate optical flow from degraded (LQ) videos.

Example:
    python inference.py --input path/to/video.mp4 --output results/demo --save_vis
"""
import argparse
import os

import cv2
import numpy as np
import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler

from daflow.flow import build_daflow, use_batch_statistics
from daflow.models import SD3Transformer2DModel
from daflow.pipeline import DAFlowPipeline
from daflow.utils.flow_viz import flow_to_image
from daflow.utils.io import frames_to_tensor, list_inputs, read_frames, write_flo, write_flow_png


def parse_args():
    p = argparse.ArgumentParser(description="DA-Flow inference")
    p.add_argument("--input", required=True, help="Video file, frame directory, or a directory of videos / frame directories.")
    p.add_argument("--output", required=True)
    p.add_argument("--sd3_path", default="preset/models/stable-diffusion-3.5-medium", help="SD3.5-medium (VAE and scheduler).")
    p.add_argument("--daflow_path", default="checkpoints/DA-Flow", help="Directory with transformer/ and daflow_flow.pth.")
    p.add_argument("--flow_ckpt", default=None, help="Flow network checkpoint (default: <daflow_path>/daflow_flow.pth).")
    p.add_argument("--null_prompt_dir", default="preset")
    # Inference settings (defaults reproduce the paper)
    p.add_argument("--num_inference_steps", type=int, default=10)
    p.add_argument("--flow_step", type=int, nargs="+", default=[1], help="Denoising step index(es) to output. The paper uses 1.")
    p.add_argument("--feature_layers", type=int, nargs="+", default=[3, 13, 16, 17])
    p.add_argument("--n_frames", type=int, default=3, help="Frames per clip in the lifted model.")
    p.add_argument("--video_chunk_size", type=int, default=31, help="Max frames per pipeline call.")
    p.add_argument("--video_chunk_overlap", type=int, default=1)
    p.add_argument("--latent_tiled_size", type=int, default=64)
    p.add_argument("--latent_tiled_overlap", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    # Outputs
    p.add_argument("--save_format", choices=["npy", "flo", "png"], default="npy")
    p.add_argument("--save_vis", action="store_true", help="Also save color-coded flow visualizations.")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=-1)
    return p.parse_args()


def build_pipeline(args, device):
    dtype = torch.float16
    vae = AutoencoderKL.from_pretrained(args.sd3_path, subfolder="vae").to(device, dtype)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.sd3_path, subfolder="scheduler")
    transformer = SD3Transformer2DModel.from_pretrained(args.daflow_path, subfolder="transformer").to(device, dtype)
    transformer.target_lifting_layer = list(range(transformer.config.num_layers))
    flow_ckpt = args.flow_ckpt or os.path.join(args.daflow_path, "daflow_flow.pth")
    flow_net = build_daflow(flow_ckpt, mixed_precision=True).to(device).eval()
    use_batch_statistics(flow_net)  # BatchNorm with batch statistics, as in training and the paper's evaluation
    for m in (vae, transformer, flow_net):
        m.requires_grad_(False)

    # Embedding of the empty prompt, identical to the one used in stage-2 training
    prompt_embeds = torch.load(os.path.join(args.null_prompt_dir, "NULL_prompt_embeds.pt"), map_location="cpu")
    pooled = torch.load(os.path.join(args.null_prompt_dir, "NULL_pooled_prompt_embeds.pt"), map_location="cpu")
    return DAFlowPipeline(
        vae, transformer, flow_net, scheduler,
        prompt_embeds=prompt_embeds[None].to(device, dtype),
        pooled_prompt_embeds=pooled[None].to(device, dtype),
        feature_layers=args.feature_layers,
        n_frames=args.n_frames,
    )


def chunk_ranges(num_frames, chunk_size, overlap, min_size):
    """Overlapping [start, end) frame ranges; the last chunk is extended to at least ``min_size`` frames."""
    chunk_size = max(chunk_size, min_size)
    if chunk_size >= num_frames:
        return [(0, num_frames)]
    overlap = min(overlap, chunk_size - 1)
    stride = chunk_size - overlap
    spans, start = [], 0
    while start < num_frames:
        end = min(start + chunk_size, num_frames)
        spans.append((start, end))
        if end == num_frames:
            break
        start += stride
    if spans[-1][1] - spans[-1][0] < min_size:
        spans[-1] = (max(0, num_frames - min_size), num_frames)
    return list(dict.fromkeys(spans))


@torch.no_grad()
def estimate_video_flow(pipe, frames, args, video_idx):
    """Forward flow of every consecutive frame pair, {step: (F-1, 2, H, W)} at the padded resolution."""
    device = pipe.device
    x = frames_to_tensor(frames).to(device)
    num_frames, _, H, W = x.shape
    if num_frames < args.n_frames:
        raise ValueError(f"DA-Flow needs at least {args.n_frames} frames, got {num_frames}.")

    torch.manual_seed(args.seed + video_idx)  # VAE posterior sampling
    generator = torch.Generator(device=device).manual_seed(args.seed + video_idx)

    flows = {}
    for start, end in chunk_ranges(num_frames, args.video_chunk_size, args.video_chunk_overlap, args.n_frames):
        noise = torch.randn((end - start, 16, H // 8, W // 8), generator=generator, device=device)
        with torch.autocast("cuda"):
            out = pipe(
                x[start:end], noise,
                num_inference_steps=args.num_inference_steps,
                flow_steps=args.flow_step,
                latent_tiled_size=args.latent_tiled_size,
                latent_tiled_overlap=args.latent_tiled_overlap,
            )
        for step, flow in out.items():
            step_flows = flows.setdefault(step, [None] * (num_frames - 1))
            for k in range(flow.shape[0]):
                if step_flows[start + k] is None:
                    step_flows[start + k] = flow[k]
    return {s: torch.stack(f, 0) for s, f in flows.items()}


def save_flows(flows, out_dir, height, width, args):
    for step, flow in flows.items():
        step_dir = out_dir if len(flows) == 1 else os.path.join(out_dir, f"step_{step:02d}")
        os.makedirs(step_dir, exist_ok=True)
        flow = flow[:, :, :height, :width].permute(0, 2, 3, 1).numpy()  # (F-1, H, W, 2)
        for i, f in enumerate(flow):
            name = os.path.join(step_dir, f"flow_{i:03d}")
            if args.save_format == "npy":
                np.save(name + ".npy", f)  # (H, W, 2)
            elif args.save_format == "flo":
                write_flo(name + ".flo", f)
            else:
                write_flow_png(name + ".png", f)
            if args.save_vis:
                os.makedirs(os.path.join(step_dir, "vis"), exist_ok=True)
                cv2.imwrite(os.path.join(step_dir, "vis", f"flow_{i:03d}.png"), flow_to_image(f)[:, :, ::-1])


def main():
    args = parse_args()
    device = torch.device("cuda")
    pipe = build_pipeline(args, device)

    inputs = list_inputs(args.input)
    end = len(inputs) if args.end_idx < 0 else args.end_idx
    for video_idx, (name, path) in enumerate(inputs):
        if not (args.start_idx <= video_idx < end):
            continue
        frames = read_frames(path)
        print(f"[{video_idx + 1}/{len(inputs)}] {name}: {len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]}")
        flows = estimate_video_flow(pipe, frames, args, video_idx)
        save_flows(flows, os.path.join(args.output, name), frames[0].shape[0], frames[0].shape[1], args)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
