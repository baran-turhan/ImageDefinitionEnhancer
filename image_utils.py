import base64

import cv2
import numpy as np


def decode_base64_image(image_base64: str):
    image_bytes = base64.b64decode(image_base64)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Görüntü okunamadı.")
    return img


def encode_image_base64(img) -> str:
    retval, buffer = cv2.imencode(".png", img)
    if not retval:
        raise ValueError("Görüntü kodlanamadı.")
    return base64.b64encode(buffer).decode("utf-8")
