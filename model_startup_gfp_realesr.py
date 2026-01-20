import os
import ssl
import torch
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# Sertifika hatası için
ssl._create_default_https_context = ssl._create_unverified_context

# --- AYARLAR ---
DEVICE = "cpu"
MODEL_PATH = os.path.join("weights", "GFPGANv1.4.pth")
BG_UPSAMPLER_MODEL_PATH = os.path.join("weights", "RealESRGAN_x4plus.pth")

restorer = None


def load_model() -> None:
    global restorer
    print(f"Model yükleniyor... Cihaz: {DEVICE}")

    if not os.path.exists(MODEL_PATH):
        print(f"Model dosyası ({MODEL_PATH}) bulunamadı")
        restorer = None
        return
    if not os.path.exists(BG_UPSAMPLER_MODEL_PATH):
        print(f"BG upsampler modeli ({BG_UPSAMPLER_MODEL_PATH}) bulunamadı")
        restorer = None
        return

    try:
        bg_model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        bg_upsampler = RealESRGANer(
            scale=4,
            model=bg_model,
            model_path=BG_UPSAMPLER_MODEL_PATH,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False,
            device=torch.device(DEVICE),
        )
        restorer = GFPGANer(
            model_path=MODEL_PATH,
            upscale=4,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=bg_upsampler,
            device=torch.device(DEVICE),
        )
        print("Model hazır")
    except Exception as e:
        print(f"Model yükleme hatası: {e}")
        restorer = None


def get_restorer():
    return restorer
