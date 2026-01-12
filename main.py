import cv2
import numpy as np
import base64
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gfpgan import GFPGANer
import os
import ssl

# Sertifika hatası için
ssl._create_default_https_context = ssl._create_unverified_context
# -------------------------------------------------------------

# --- AYARLAR ---
DEVICE = 'cpu'

# Model ayarları
MODEL_PATH = os.path.join("gfpgan", "weights", "GFPGANv1.4.pth")

app = FastAPI(title="Face Restorer API")
restorer = None

@app.on_event("startup")
def load_model():
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
            arch='clean', 
            channel_multiplier=2, 
            bg_upsampler=None,
            device=torch.device(DEVICE)
        )
        print("✅ GFPGAN Modeli Hazır!")
    except Exception as e:
        print(f"❌ Model yükleme hatası: {e}")
        restorer = None

class ImageRequest(BaseModel):
    image_base64: str

class ImageResponse(BaseModel):
    restored_base64: str

@app.post("/restore", response_model=ImageResponse)
async def restore_face(payload: ImageRequest):
    if restorer is None:
        raise HTTPException(status_code=500, detail="Model yüklü değil.")

    try:
        # 1. Base64 -> Image
        image_bytes = base64.b64decode(payload.image_base64)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Görüntü okunamadı.")

        # 2. GFPGAN
        _, _, restored_img = restorer.enhance(
            img, 
            has_aligned=False, 
            only_center_face=False, 
            paste_back=True
        )

        # 3. Image -> Base64
        retval, buffer = cv2.imencode('.jpg', restored_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        base64_output = base64.b64encode(buffer).decode('utf-8')

        return {"restored_base64": base64_output}

    except Exception as e:
        print(f"Hata detayı: {e}")
        raise HTTPException(status_code=500, detail=str(e))