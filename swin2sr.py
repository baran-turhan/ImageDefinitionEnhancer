#!/usr/bin/env python3
import argparse
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Run Swin2SR super-resolution on an image.")
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument("--out", type=str, default="outputs/restored.png")
    parser.add_argument(
        "--model",
        type=str,
        default="caidas/swin2sr-realworld-sr-x4-64-bsrgan-psnr",
        help="Hugging Face model id",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    device = "mps"

    try:
        import torch
        from PIL import Image
        from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor
    except Exception as exc:
        print(
            "Eksik kütüphaneler. Şunu çalıştırın:\n"
            "  pip install torch transformers pillow\n"
            f"Hata: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        image = Image.open(args.input).convert("RGB")
    except IOError:
        print(f"Hata: Görüntü dosyası açılamadı: {args.input}")
        sys.exit(1)

    processor = Swin2SRImageProcessor()
    model = Swin2SRForImageSuperResolution.from_pretrained(
        args.model
    ).to(device)

    inputs = processor(image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    sr = outputs.reconstruction
    sr = sr.squeeze(0).clamp(0, 1).cpu()
    sr = (sr * 255.0).round().byte().permute(1, 2, 0).numpy()
    sr_image = Image.fromarray(sr)

    output_dir = os.path.dirname(args.out)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    sr_image.save(args.out)
    print(f"Başarıyla kaydedildi: {args.out}")

if __name__ == "__main__":
    main()