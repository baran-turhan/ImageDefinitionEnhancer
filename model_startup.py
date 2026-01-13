import os
import ssl
import torch
from gfpgan import GFPGANer

# Sertifika hatası için
ssl._create_default_https_context = ssl._create_unverified_context

# --- AYARLAR ---
DEVICE = "cpu"
MODEL_PATH = os.path.join("models", "weights", "GFPGANv1.4.pth")

restorer = None


def load_model() -> None:
    global restorer
    print(f"Model yükleniyor... Cihaz: {DEVICE}")

    if not os.path.exists(MODEL_PATH):
        print(f"Model dosyası ({MODEL_PATH}) bulunamadı")
        restorer = None
        return

    try:
        restorer = GFPGANer(
            model_path=MODEL_PATH,
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device=torch.device(DEVICE),
        )
        print("Model hazır")
    except Exception as e:
        print(f"Model yükleme hatası: {e}")
        restorer = None


def get_restorer():
    return restorer
