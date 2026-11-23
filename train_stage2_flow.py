"""Stage 2: train the DA-Flow network on features of the frozen lifted transformer.

For each clip of F frames, the lifted DiT4SR transformer (stage 1) is run once at a
random noise level. Query features of frame i and key features of frame j (i < j)
are taken from the selected full-attention layers and, together with the degraded
frames, fed to the DA-Flow network. Supervision is SEA-RAFT flow computed on the HQ
frames (pseudo ground truth); inference only needs the degraded frames.
"""
import argparse
import copy
import logging
import math
import os
import re

import diffusers
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, free_memory
from einops import rearrange, repeat
from tqdm.auto import tqdm

from daflow.data.youhq import VideoPairedCaptionDataset, collate_fn
from daflow.flow import build_daflow, sequence_loss
from daflow.models import SD3Transformer2DModel
from daflow.sea_raft import SeaRaftFlowExtractor
from daflow.utils.training import find_resume_checkpoint, prune_checkpoints

logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="DA-Flow stage 2: flow network training")
    # Models / data
    p.add_argument("--pretrained_model_name_or_path", default="preset/models/stable-diffusion-3.5-medium")
    p.add_argument("--transformer_model_name_or_path", required=True, help="Stage-1 checkpoint dir (contains transformer/).")
    p.add_argument("--raft_model_path", default="preset/models/raft-things.pth", help="RAFT initialization.")
    p.add_argument("--sea_raft_model_path", default="preset/models/sea_raft/model.safetensors", help="SEA-RAFT for pseudo-GT flow.")
    p.add_argument("--null_prompt_dir", default="preset")
    p.add_argument("--train_root", required=True, help="HQ frame root, e.g. data/YouHQ/YouHQ-Train-frames")
    p.add_argument("--lq_suffix", default="-LQ-frames")
    p.add_argument("--output_dir", default="experiments/stage2_flow")
    # DA-Flow
    p.add_argument("--n_frames", type=int, default=3)
    p.add_argument("--lifting_layers", type=int, nargs="+", default=list(range(24)))
    p.add_argument("--feature_layers", type=int, nargs="+", default=[3, 13, 16, 17], help="Layers whose query/key features are used.")
    p.add_argument("--flow_iters", type=int, default=12)
    p.add_argument("--gamma", type=float, default=0.8, help="Sequence-loss decay.")
    p.add_argument("--sea_raft_iters", type=int, default=20)
    # Optimization
    p.add_argument("--train_batch_size", type=int, default=4, help="Clips per GPU.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--max_train_steps", type=int, default=20000)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--lr_scheduler", default="constant")
    p.add_argument("--lr_warmup_steps", type=int, default=500)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--adam_weight_decay", type=float, default=1e-2)
    p.add_argument("--adam_epsilon", type=float, default=1e-8)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--weighting_scheme", default="logit_normal", choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"])
    p.add_argument("--logit_mean", type=float, default=0.0)
    p.add_argument("--logit_std", type=float, default=1.0)
    p.add_argument("--mode_scale", type=float, default=1.29)
    p.add_argument("--mixed_precision", default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--allow_tf32", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    # Logging / checkpointing
    p.add_argument("--report_to", default="tensorboard", help="tensorboard | wandb | all | none")
    p.add_argument("--tracker_project_name", default="DA-Flow")
    p.add_argument("--checkpointing_steps", type=int, default=5000)
    p.add_argument("--checkpoints_total_limit", type=int, default=5)
    p.add_argument("--resume_from_checkpoint", default=None, help="Checkpoint dir or 'latest'.")
    return p.parse_args()


def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None if args.report_to == "none" else args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, "logs")),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_warning()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(accelerator.mixed_precision, torch.float32)

    # Frozen models
    noise_scheduler_copy = copy.deepcopy(FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler"))
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    transformer = SD3Transformer2DModel.from_pretrained_local(args.transformer_model_name_or_path, subfolder="transformer")
    transformer.target_lifting_layer = args.lifting_layers
    transformer.lifting_n_frames = args.n_frames
    flow_extractor = SeaRaftFlowExtractor(args.sea_raft_model_path, iters=args.sea_raft_iters)
    for m in (vae, transformer, flow_extractor):
        m.requires_grad_(False)
    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    flow_extractor.to(accelerator.device)

    # Empty-prompt conditioning (no text encoders needed)
    null_embeds = torch.load(os.path.join(args.null_prompt_dir, "NULL_prompt_embeds.pt"), map_location="cpu").to(accelerator.device)
    null_pooled = torch.load(os.path.join(args.null_prompt_dir, "NULL_pooled_prompt_embeds.pt"), map_location="cpu").to(accelerator.device)

    # Hooks on query / key projections. Each projection runs twice per block (image
    # stream, then LQ control stream): entries 4l and 4l+1 are the image-stream q / k.
    features = []
    pattern = re.compile(r"transformer_blocks\.(\d+)\.attn\.(to_q|to_k)$")
    for name, module in transformer.named_modules():
        m = pattern.match(name)
        if m and int(m.group(1)) in args.feature_layers:
            module.register_forward_hook(lambda _m, _i, out: features.append(out))

    # Trainable DA-Flow network, initialized from RAFT (C+T)
    flow_net = build_daflow(args.raft_model_path, from_raft=True, mixed_precision=args.mixed_precision != "no")
    flow_net.to(accelerator.device, dtype=torch.float32)
    flow_net.train()
    params_to_optimize = [p for p in flow_net.parameters() if p.requires_grad]
    logger.info(f"Trainable DA-Flow parameters: {sum(p.numel() for p in params_to_optimize):,}")

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(output_dir, "daflow_flow.pth"))
            weights.clear()

    def load_model_hook(models, input_dir):
        model = models.pop()
        model.load_state_dict(torch.load(os.path.join(input_dir, "daflow_flow.pth"), map_location="cpu"))

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
                                  weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    train_dataset = VideoPairedCaptionDataset(args.train_root, lq_suffix=args.lq_suffix, null_text_ratio=1.0, n_frames=args.n_frames)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size,
                                                   num_workers=args.dataloader_num_workers, collate_fn=collate_fn)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
                                 num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                 num_training_steps=args.max_train_steps * accelerator.num_processes)
    flow_net, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(flow_net, optimizer, train_dataloader, lr_scheduler)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if accelerator.is_main_process and args.report_to != "none":
        accelerator.init_trackers(args.tracker_project_name, config={k: str(v) if isinstance(v, list) else v for k, v in vars(args).items()})

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running stage 2 (flow) *****")
    logger.info(f"  Num clips = {len(train_dataset)}, frames per clip = {args.n_frames}, feature layers = {args.feature_layers}")
    logger.info(f"  Total batch size = {total_batch_size}, optimization steps = {args.max_train_steps}")

    global_step, first_epoch = 0, 0
    resume_path = find_resume_checkpoint(args.output_dir, args.resume_from_checkpoint)
    if resume_path is not None:
        accelerator.print(f"Resuming from {resume_path}")
        accelerator.load_state(resume_path)
        global_step = int(os.path.basename(resume_path).split("-")[1])
        first_epoch = global_step // num_update_steps_per_epoch
    progress_bar = tqdm(range(args.max_train_steps), initial=global_step, desc="Steps", disable=not accelerator.is_local_main_process)

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps.to(accelerator.device)]
        sigma = sigmas[step_indices].flatten()
        while sigma.ndim < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    free_memory()
    F_ = args.n_frames
    for epoch in range(first_epoch, num_train_epochs):
        for batch in train_dataloader:
            with accelerator.accumulate(flow_net):
                with torch.no_grad():
                    model_input = vae.encode(batch["gt_image"].to(dtype=weight_dtype)).latent_dist.sample()
                    model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
                    control = vae.encode(batch["lq_image"].to(dtype=weight_dtype)).latent_dist.sample()
                    control = (control - vae.config.shift_factor) * vae.config.scaling_factor

                    # Noisy latent at a random timestep (shared by the frames of a clip)
                    noise = torch.randn_like(model_input)
                    bsz = model_input.shape[0] // F_
                    u = compute_density_for_timestep_sampling(weighting_scheme=args.weighting_scheme, batch_size=bsz,
                                                              logit_mean=args.logit_mean, logit_std=args.logit_std, mode_scale=args.mode_scale)
                    u = repeat(u, "b -> (b f)", f=F_)
                    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                    timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
                    sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                    noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                    features.clear()
                    transformer(
                        hidden_states=noisy_model_input,
                        controlnet_image=control,
                        timestep=timesteps,
                        encoder_hidden_states=null_embeds[None].repeat(bsz * F_, 1, 1),
                        pooled_projections=null_pooled[None].repeat(bsz * F_, 1),
                        return_dict=False,
                    )

                # Query features of frame i and key features of frame j (i < j)
                idx1 = torch.tensor(batch["flow1_idx"], device=accelerator.device)
                idx2 = torch.tensor(batch["flow2_idx"], device=accelerator.device)
                ar = torch.arange(bsz, device=accelerator.device)
                q_feats, k_feats = [], []
                for l in range(len(args.feature_layers)):
                    q = features[4 * l].view(bsz, F_, -1, features[4 * l].shape[-1])
                    k = features[4 * l + 1].view(bsz, F_, -1, features[4 * l + 1].shape[-1])
                    q_feats.append(q[ar, idx1])
                    k_feats.append(k[ar, idx2])

                lq = rearrange(batch["lq_image"].to(dtype=weight_dtype), "(b f) c h w -> b f c h w", f=F_)
                gt = rearrange(batch["gt_image"].to(dtype=weight_dtype), "(b f) c h w -> b f c h w", f=F_)
                lq1, lq2 = lq[ar, idx1], lq[ar, idx2]
                flow_gt = flow_extractor(gt[ar, idx1], gt[ar, idx2])  # SEA-RAFT pseudo-GT on HQ frames

                flow_preds = flow_net(lq1, lq2, [q_feats, k_feats], iters=args.flow_iters)
                loss, metrics = sequence_loss(flow_preds, flow_gt, torch.ones_like(flow_gt[:, :1]), args.gamma)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                logs = {"train/flow_loss": loss.detach().item(), "train/lr": lr_scheduler.get_last_lr()[0]}
                logs.update({f"train/{k}": float(v) for k, v in metrics.items()})
                progress_bar.set_postfix(loss=logs["train/flow_loss"], epe=logs["train/epe"])
                accelerator.log(logs, step=global_step)
                if accelerator.is_main_process and (global_step % args.checkpointing_steps == 0 or global_step == args.max_train_steps):
                    prune_checkpoints(args.output_dir, args.checkpoints_total_limit, logger)
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")
            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
