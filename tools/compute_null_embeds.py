"""Precompute SD3 embeddings of the empty prompt (CLIP 77 + T5 77 tokens).

DA-Flow never uses text in stage 2 or at inference: the transformer is conditioned on
the empty prompt. The files shipped in ``preset/`` are the ones used for training;
regenerated embeddings can differ slightly in numerical precision.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from daflow.utils.text import encode_prompt, load_text_encoders

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sd3_path", default="preset/models/stable-diffusion-3.5-medium")
    p.add_argument("--output_dir", default="preset")
    p.add_argument("--max_sequence_length", type=int, default=77)
    args = p.parse_args()
    tokenizers, encoders = load_text_encoders(args.sd3_path, "cuda", torch.float16)
    with torch.autocast("cuda"):
        embeds, pooled = encode_prompt(encoders, tokenizers, [""], max_sequence_length=args.max_sequence_length)
    torch.save(embeds[0].cpu(), os.path.join(args.output_dir, "NULL_prompt_embeds.pt"))
    torch.save(pooled[0].cpu(), os.path.join(args.output_dir, "NULL_pooled_prompt_embeds.pt"))
    print(tuple(embeds[0].shape), tuple(pooled[0].shape))
