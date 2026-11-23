"""SD3 text encoding (CLIP-L + CLIP-G + T5-XXL), following the diffusers SD3 DreamBooth example."""
import torch
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast


def load_text_encoders(sd3_path, device, dtype):
    tokenizers = [
        CLIPTokenizer.from_pretrained(sd3_path, subfolder="tokenizer"),
        CLIPTokenizer.from_pretrained(sd3_path, subfolder="tokenizer_2"),
        T5TokenizerFast.from_pretrained(sd3_path, subfolder="tokenizer_3"),
    ]
    encoders = [
        CLIPTextModelWithProjection.from_pretrained(sd3_path, subfolder="text_encoder"),
        CLIPTextModelWithProjection.from_pretrained(sd3_path, subfolder="text_encoder_2"),
        T5EncoderModel.from_pretrained(sd3_path, subfolder="text_encoder_3"),
    ]
    for e in encoders:
        e.requires_grad_(False)
        e.to(device, dtype=dtype)
    return tokenizers, encoders


def _encode_clip(encoder, tokenizer, prompt, device):
    ids = tokenizer(prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
    out = encoder(ids.to(device), output_hidden_states=True)
    return out.hidden_states[-2].to(encoder.dtype), out[0]


def _encode_t5(encoder, tokenizer, prompt, max_sequence_length, device):
    ids = tokenizer(prompt, padding="max_length", max_length=max_sequence_length, truncation=True,
                    add_special_tokens=True, return_tensors="pt").input_ids
    return encoder(ids.to(device))[0].to(encoder.dtype)


@torch.no_grad()
def encode_prompt(encoders, tokenizers, prompt, max_sequence_length=77, device=None):
    """Returns prompt_embeds (B, 77 + max_sequence_length, 4096) and pooled_prompt_embeds (B, 2048)."""
    prompt = [prompt] if isinstance(prompt, str) else prompt
    device = device or encoders[0].device
    clip_embeds, pooled = [], []
    for enc, tok in zip(encoders[:2], tokenizers[:2]):
        e, p = _encode_clip(enc, tok, prompt, device)
        clip_embeds.append(e)
        pooled.append(p)
    clip_embeds = torch.cat(clip_embeds, dim=-1)
    pooled = torch.cat(pooled, dim=-1)
    t5_embeds = _encode_t5(encoders[2], tokenizers[2], prompt, max_sequence_length, device)
    clip_embeds = torch.nn.functional.pad(clip_embeds, (0, t5_embeds.shape[-1] - clip_embeds.shape[-1]))
    return torch.cat([clip_embeds, t5_embeds], dim=-2), pooled
