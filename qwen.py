#!/usr/bin/env python3
import argparse
import os
import sys
from PIL import Image

MAX_INPUT_SIZE = 96

def resize(img, max_size=MAX_INPUT_SIZE):
    w, h = img.size
    if max(w, h) <= max_size:
        print(f"[i] Input size OK: {w}x{h}")
        return img

    scale = max_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    print(f"[i] Resizing input: {w}x{h} → {new_w}x{new_h}")
    return img.resize((new_w, new_h), Image.LANCZOS)

def parse_args():
    parser = argparse.ArgumentParser(description="Qwen-Image-Edit-2511 image-to-image edit.")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--init_image", type=str, required=True, help="Path or URL to input image")
    parser.add_argument("--out", type=str, default="outputs_qwen_edit/qwen_edit.png")
    parser.add_argument("--model", type=str, default="Qwen/Qwen-Image-Edit-2511")
    parser.add_argument("--lora", type=str, default="prithivMLmods/Qwen-Image-Edit-2511-Unblur-Upscale")
    return parser.parse_args()

def main():
    args = parse_args()

    try:
        import torch
        from diffusers import DiffusionPipeline
        from diffusers.utils import load_image
    except Exception as exc:
        print(
            "Missing dependencies. Install with:\n"
            "  python -m pip install diffusers transformers accelerate safetensors\n"
            "Error: {}".format(exc),
            file=sys.stderr,
        )
        sys.exit(1)

    pipe = DiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map="mps",
    )
    if args.lora:
        pipe.load_lora_weights(args.lora)

    input_image = load_image(args.init_image)
    proper_image = resize(input_image)
    result = pipe(image=proper_image, prompt=args.prompt)
    image = result.images[0]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    image.save(args.out)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
