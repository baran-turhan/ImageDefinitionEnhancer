import argparse
import torch
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image

MAX_INPUT_SIZE = 180

def load_image(path):
    return Image.open(path).convert("RGB")

def resize_if_needed(img, max_size=180):
    w, h = img.size
    if max(w, h) <= max_size:
        print(f"[i] Input size OK: {w}x{h}")
        return img

    scale = max_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    print(f"[i] Resizing input: {w}x{h} → {new_w}x{new_h}")
    return img.resize((new_w, new_h), Image.LANCZOS)

def upscale_image(pipe, img, prompt="", steps=30):
    result = pipe(prompt=prompt, image=img, num_inference_steps=steps)
    return result.images[0]

def main(args):
    device = "mps"
    low_res_img = load_image(args.input)
    low_res_img = resize_if_needed(low_res_img, MAX_INPUT_SIZE)

    print(f"[+] Loading Stable Diffusion x4 Upscaler on {device.upper()}…")
    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler",
        torch_dtype=torch.float32
    )
    pipe = pipe.to(device)

    pipe.enable_attention_slicing()

    print("[+] Upscaling… This can take a while on CPU/MPS!")
    upscaled = upscale_image(pipe, low_res_img, prompt=args.prompt, steps=args.steps)

    print(f"[+] Saving output to {args.output}")
    upscaled.save(args.output)
    print("[✔] Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stable Diffusion x4 Upscaler")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to low-res image")
    parser.add_argument("--output", "-o", type=str, default="upscaled.png", help="Save path")
    parser.add_argument("--prompt", "-p", type=str, default="", help="Upscaler prompt (optional)")
    parser.add_argument("--steps", "-s", type=int, default=30, help="Inference steps (quality vs. time)")
    main(parser.parse_args())
