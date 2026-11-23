"""Stage 1: lift the DiT4SR image restoration model to video.

Every MM-DiT block attends jointly over the tokens of F consecutive frames (full
spatio-temporal attention). Only the attention layers are finetuned with the
flow-matching diffusion loss on paired HQ / degraded YouHQ clips.
"""
import argparse
import copy
import logging
import math
import os

import diffusers
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, free_memory
from einops import repeat
from tqdm.auto import tqdm

from daflow.data.youhq import VideoPairedCaptionDataset, collate_fn
from daflow.models import SD3Transformer2DModel
from daflow.utils.text import encode_prompt, load_text_encoders
from daflow.utils.training import find_resume_checkpoint, prune_checkpoints

logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="DA-Flow stage 1: lifting DiT4SR to video")
    # Models / data
    p.add_argument("--pretrained_model_name_or_path", default="preset/models/stable-diffusion-3.5-medium")
    p.add_argument("--transformer_model_name_or_path", default="preset/models/dit4sr_q", help="DiT4SR checkpoint (contains transformer/).")
    p.add_argument("--train_root", required=True, help="HQ frame root, e.g. data/YouHQ/YouHQ-Train-frames")
    p.add_argument("--lq_suffix", default="-LQ-frames", help="LQ frame root = train_root with '-frames' replaced by this suffix.")
    p.add_argument("--output_dir", default="experiments/stage1_lifting")
    # Lifting
    p.add_argument("--n_frames", type=int, default=3)
    p.add_argument("--lifting_layers", type=int, nargs="+", default=list(range(24)), help="Blocks with full spatio-temporal attention (trained).")
    p.add_argument("--null_text_ratio", type=float, default=0.2)
    p.add_argument("--max_sequence_length", type=int, default=77)
    # Optimization
    p.add_argument("--train_batch_size", type=int, default=1, help="Clips per GPU.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
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
    p.add_argument("--precondition_outputs", type=int, default=1)
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

    # Models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae.requires_grad_(False)
    tokenizers, text_encoders = load_text_encoders(args.pretrained_model_name_or_path, accelerator.device, weight_dtype)

    transformer = SD3Transformer2DModel.from_pretrained_local(args.transformer_model_name_or_path, subfolder="transformer")
    transformer.requires_grad_(False)
    transformer.target_lifting_layer = args.lifting_layers
    transformer.lifting_n_frames = args.n_frames
    trainable_modules = tuple(f"transformer_blocks.{l}.attn" for l in args.lifting_layers)
    for name, param in transformer.named_parameters():
        if any(m in name for m in trainable_modules):
            param.requires_grad = True

    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    if args.mixed_precision == "fp16":
        cast_training_params([transformer], dtype=torch.float32)  # trainable params in fp32
    params_to_optimize = [p for p in transformer.parameters() if p.requires_grad]
    logger.info(f"Trainable transformer parameters: {sum(p.numel() for p in params_to_optimize):,}")

    # Save / load only the transformer in diffusers format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                accelerator.unwrap_model(model).save_pretrained(os.path.join(output_dir, "transformer"))
            weights.clear()

    def load_model_hook(models, input_dir):
        model = models.pop()
        loaded = SD3Transformer2DModel.from_pretrained(input_dir, subfolder="transformer")
        model.register_to_config(**loaded.config)
        model.load_state_dict(loaded.state_dict())
        del loaded

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
                                  weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)

    train_dataset = VideoPairedCaptionDataset(args.train_root, lq_suffix=args.lq_suffix, null_text_ratio=args.null_text_ratio, n_frames=args.n_frames)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size,
                                                   num_workers=args.dataloader_num_workers, collate_fn=collate_fn)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
                                 num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                 num_training_steps=args.max_train_steps * accelerator.num_processes)
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(transformer, optimizer, train_dataloader, lr_scheduler)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if accelerator.is_main_process and args.report_to != "none":
        accelerator.init_trackers(args.tracker_project_name, config={k: str(v) if isinstance(v, list) else v for k, v in vars(args).items()})

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running stage 1 (lifting) *****")
    logger.info(f"  Num clips = {len(train_dataset)}, frames per clip = {args.n_frames}")
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
    for epoch in range(first_epoch, num_train_epochs):
        for batch in train_dataloader:
            with accelerator.accumulate(transformer):
                with torch.no_grad():
                    model_input = vae.encode(batch["gt_image"].to(dtype=weight_dtype)).latent_dist.sample()
                    model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
                    control = vae.encode(batch["lq_image"].to(dtype=weight_dtype)).latent_dist.sample()
                    control = (control - vae.config.shift_factor) * vae.config.scaling_factor

                # One timestep per clip, shared by its frames
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0] // args.n_frames
                u = compute_density_for_timestep_sampling(weighting_scheme=args.weighting_scheme, batch_size=bsz,
                                                          logit_mean=args.logit_mean, logit_std=args.logit_std, mode_scale=args.mode_scale)
                u = repeat(u, "b -> (b f)", f=args.n_frames)
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, batch["prompt"], args.max_sequence_length)
                prompt_embeds = repeat(prompt_embeds, "b ... -> (b f) ...", f=args.n_frames)
                pooled_prompt_embeds = repeat(pooled_prompt_embeds, "b ... -> (b f) ...", f=args.n_frames)

                model_pred = transformer(
                    hidden_states=noisy_model_input,
                    controlnet_image=control,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]
                if args.precondition_outputs:
                    model_pred = model_pred * (-sigmas) + noisy_model_input
                    target = model_input
                else:
                    target = noise - model_input
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                loss = torch.mean((weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), 1).mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                logs = {"train/diff_loss": loss.detach().item(), "train/lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(loss=logs["train/diff_loss"])
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
