# AnovaGreen Field Diagnosis - Local MVP Simulator

A local web app that runs the best trained ViT model for plant disease classification. Upload a leaf image and get top-3 disease predictions.

## Prerequisites

- Python 3.10+
- Node.js 18+
- The model checkpoint (not committed to git due to size)

## Quick Start

### 0. Model Setup (required)

The model checkpoint is **not included in the repo** (gitignored). You must obtain it and place it at:

```
results_zip/best_model/erasing_adamw_vit_base_patch16_224.pt
```

See [`backend/assets/model_manifest.json`](backend/assets/model_manifest.json) for the exact filename, expected path, and SHA256 hash to verify integrity.

To verify your checkpoint:

```bash
# Linux / macOS / Git Bash (Windows)
sha256sum results_zip/best_model/erasing_adamw_vit_base_patch16_224.pt
```

Expected: `5776de5116ddbff02678e89ce52b2510e4fc50352729def449c81ae1abab2682`

### 1. Backend (FastAPI)

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

First startup takes ~10-20 seconds to load the 550 MB model into CPU memory.

Verify: http://localhost:8000/health

### 2. Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
```

Opens at http://localhost:5173

### 3. Smoke Test (optional)

```bash
pip install requests
python backend/scripts/smoke_test_predict.py path/to/leaf.jpg
```

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Backend status and model info |
| `/predict` | POST | Upload image, get top-k predictions |

### POST /predict

- **Body:** `multipart/form-data` with field `file` (JPEG/PNG/WebP)
- **Query params:** `top_k` (int, default 3), `threshold` (float, default 0.5)
- **Response:**
```json
{
  "model_version": "vit_base_patch16_224 | erasing_adamw | epoch=19 | val_acc=0.9987",
  "top1": { "label": "tomato__late_blight", "prob": 0.63 },
  "topk": [
    { "rank": 1, "label": "tomato__late_blight", "prob": 0.63 },
    { "rank": 2, "label": "tomato__early_blight", "prob": 0.18 },
    { "rank": 3, "label": "tomato__healthy", "prob": 0.09 }
  ],
  "warning": null
}
```

## Architecture

```
backend/
  app/main.py          FastAPI app (CORS, /health, /predict)
  app/inference.py     Model loading + prediction (reuses src/utils/baseline_models.py)
  assets/classes.json         26 class labels in index order
  assets/model_manifest.json  Checkpoint filename, path, SHA256
  requirements.txt

frontend/
  src/App.jsx          Main layout
  src/components/      Header, ImagePanel, ResultPanel
  src/api/predict.js   Backend API wrapper
```

## Model Details

- **Architecture:** ViT-B/16 (timm)
- **Checkpoint:** `erasing_adamw_vit_base_patch16_224.pt`
- **Classes:** 26 plant disease categories
- **Inference:** CPU-only, ~1-3 seconds per image
