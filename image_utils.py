import base64

import cv2
import numpy as np
from PIL import Image

def load_image(path):
    return Image.open(path).convert("RGB")


def decode_base64_image(image_base64: str):
    image_bytes = base64.b64decode(image_base64)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Görüntü okunamadı.")
    return img


def encode_image_base64(img) -> str:
    if isinstance(img, Image.Image):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    retval, buffer = cv2.imencode(".png", img)
    if not retval:
        raise ValueError("Görüntü kodlanamadı.")
    return base64.b64encode(buffer).decode("utf-8")

def main():
    img = load_image("test_samples/test-TCNTAPAN.png")
    imgStr = encode_image_base64(img)
    out_path = "outputs_base64.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(imgStr)
    print(f"Saved: {out_path}")
    

if __name__ == "__main__":
    main()
