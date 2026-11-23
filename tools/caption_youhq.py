"""Caption YouHQ clips for stage-1 training.

Following DiT4SR, each clip is described by LLaVA-v1.5-13B from its first frame, and
sentences about focus / blur are removed. The captions are used as text prompts in
stage 1 (80% of the time; the empty prompt otherwise).

Output: <output>/<category>/<video>/<clip>.txt, to be passed to
``data_preparation/prepare_youhq.py --prompt_root``.

Example:
    python tools/caption_youhq.py --root /path/to/YouHQ-Train --output data/YouHQ-captions
"""
import argparse
import glob
import os
import re

import cv2
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

QUESTION = (
    "Please describe the actual objects in the image in a very detailed manner. "
    "Please do not include descriptions related to the focus and bokeh of this image. "
    "Please do not include descriptions like the background is blurred."
)
SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
PROHIBITED = ["focus", "focal", "prominent", "close-up", "black and white", "blur", "depth", "dense", "locate", "position"]


def remove_focus_sentences(text):
    parts = re.split(r"([.?!])", text)
    kept = []
    for i in range(0, len(parts), 2):
        sentence = parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")
        if not any(w in sentence.lower() for w in PROHIBITED):
            kept.append(sentence)
    return "".join(kept).strip()


def first_frame(path):
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read {path}")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="YouHQ-Train root: <category>/<video>/<clip>.mp4")
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="llava-hf/llava-1.5-13b-hf")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rank", type=int, default=0, help="Process every world_size-th clip starting at rank.")
    p.add_argument("--world_size", type=int, default=1)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    processor = AutoProcessor.from_pretrained(args.model)
    model = LlavaForConditionalGeneration.from_pretrained(args.model, dtype=torch.float16).to("cuda").eval()
    prompt = f"{SYSTEM} USER: <image>\n{QUESTION} ASSISTANT:"

    videos = sorted(glob.glob(os.path.join(args.root, "*", "*", "*.mp4")))[args.rank::args.world_size]
    for path in tqdm(videos):
        rel = os.path.splitext(os.path.relpath(path, args.root))[0]
        out_path = os.path.join(args.output, rel + ".txt")
        if os.path.exists(out_path):
            continue
        inputs = processor(images=first_frame(path), text=prompt, return_tensors="pt").to("cuda", torch.float16)
        with torch.no_grad():
            out = model.generate(**inputs, do_sample=True, temperature=0.2, top_p=0.7, max_new_tokens=512)
        text = processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        text = remove_focus_sentences(text.strip().replace("\n", " ").replace("\r", " "))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
