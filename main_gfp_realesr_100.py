#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import cv2

from model_startup_gfp_realesr import get_restorer, load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run GFPGAN on all images in a folder.")
    parser.add_argument("--input_dir", type=str, default="test_samples")
    parser.add_argument("--out_dir", type=str, default="outputs_gfp")
    parser.add_argument("--weight", type=float, default=0.1)
    return parser.parse_args()


def is_image(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def main():
    args = parse_args()
    load_model()
    restorer = get_restorer()
    if restorer is None:
        raise SystemExit("Model yuklenemedi. weights klasorunu kontrol edin.")

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = [p for p in sorted(input_dir.iterdir()) if p.is_file() and is_image(p)]
    if not images:
        raise SystemExit(f"Gecerli resim bulunamadi: {input_dir}")

    for path in images:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[skip] Okunamadi: {path.name}")
            continue

        _, _, restored_img = restorer.enhance(
            img,
            has_aligned=False,
            only_center_face=True,
            paste_back=True,
            weight=args.weight,
        )

        out_path = out_dir / f"{path.stem}_gfp.png"
        cv2.imwrite(str(out_path), restored_img)
        print(f"[ok] {path.name} -> {out_path}")


if __name__ == "__main__":
    main()
