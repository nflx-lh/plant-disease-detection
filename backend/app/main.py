"""
FastAPI application for AnovaGreen Field Diagnosis.
"""

import io
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from . import inference

app = FastAPI(title="AnovaGreen Field Diagnosis API")

# CORS: allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    """Load model on server startup."""
    inference.load_model()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": inference.is_loaded(),
        "model_version": inference.get_model_version(),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    top_k: int = Query(3, ge=1, le=26),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
):
    # Validate content type
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(400, "File must be a JPEG, PNG, or WebP image.")

    # Read and open image
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(400, "Could not decode image file.")

    # Run prediction
    result = inference.predict(image, top_k=top_k, threshold=threshold)
    return result
