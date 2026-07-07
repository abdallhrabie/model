"""
Tomato Health Check API
========================

POST /analyze
  form-data:
    - images : أكتر من ملف صورة (كل الصفوف بنفس الـ key "images")
    - source : "camera"  -> Step1 (PlantNet, اختياري) -> Step2 (قص الورق) -> Step3 (anomaly detector)
               "upload"  -> Step3 مباشرة (موديل كشف المرض) لكل صورة

الرد بيبقى متوسط نتايج كل الصور المبعوتة في نفس الطلب.

تشغيل السيرفر محليًا:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import io
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

from aggregation import aggregate_camera_results, aggregate_upload_results
from pipeline import process_camera_image, process_upload_image

app = FastAPI(title="Tomato Health Check API")


@app.post("/analyze")
async def analyze(
    images: List[UploadFile] = File(...),
    source: str = Form(...),
):
    if source not in ("camera", "upload"):
        raise HTTPException(status_code=400, detail="source لازم يكون 'camera' أو 'upload'")

    if not images:
        raise HTTPException(status_code=400, detail="لازم تبعت صورة واحدة على الأقل")

    pil_images = []
    for img_file in images:
        raw = await img_file.read()
        try:
            pil_images.append((Image.open(io.BytesIO(raw)).convert("RGB"), raw))
        except Exception:
            raise HTTPException(
                status_code=400,
                detail=f"ملف {img_file.filename} تالف أو مش صورة مدعومة",
            )

    # ================= UPLOAD PATH =================
    if source == "upload":
        results = [process_upload_image(pil_img) for pil_img, _ in pil_images]
        return {"source": "upload", **aggregate_upload_results(results)}

    # ================= CAMERA PATH =================
    results = [process_camera_image(pil_img, raw) for pil_img, raw in pil_images]
    return {"source": "camera", **aggregate_camera_results(results)}


@app.get("/health")
async def health_check():
    return {"status": "ok"}
