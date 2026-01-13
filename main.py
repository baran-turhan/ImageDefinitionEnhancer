from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from image_utils import decode_base64_image, encode_image_base64
from model_startup import get_restorer, load_model as load_gfpgan_model

app = FastAPI(title="Face Restorer API")

@app.on_event("startup")
def on_startup():
    load_gfpgan_model()

class ImageRequest(BaseModel):
    image_base64: str

class ImageResponse(BaseModel):
    restored_base64: str

@app.post("/restore", response_model=ImageResponse)
async def restore_face(payload: ImageRequest):
    restorer = get_restorer()
    if restorer is None:
        raise HTTPException(status_code=500, detail="Model yüklü değil.")

    try:
        # 1. Base64 -> Image
        img = decode_base64_image(payload.image_base64)

        # 2. GFPGAN
        _, _, restored_img = restorer.enhance(
            img, 
            has_aligned=False, 
            only_center_face=False, 
            paste_back=True
        )

        # 3. Image -> Base64
        base64_output = encode_image_base64(restored_img)

        return {"restored_base64": base64_output}

    except Exception as e:
        print(f"Hata detayı: {e}")
        raise HTTPException(status_code=500, detail=str(e))
