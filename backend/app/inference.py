"""
Inference module: loads the ViT checkpoint and runs single-image prediction.
Reuses the project's existing get_model() factory from src/utils/baseline_models.
"""

import sys
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from timm.data import resolve_model_data_config, create_transform

# Add project root to sys.path so we can import src.utils
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.baseline_models import get_model

# Module-level state (singleton)
_state = {
    "model": None,
    "transform": None,
    "class_names": None,
    "device": None,
    "model_version": None,
}

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "results_zip"
    / "best_model"
    / "erasing_adamw_vit_base_patch16_224.pt"
)
CLASSES_PATH = Path(__file__).resolve().parent.parent / "assets" / "classes.json"


def load_model():
    """Load model checkpoint and class names. Called once at startup."""
    device = torch.device("cpu")

    # Load class names
    with open(CLASSES_PATH, "r") as f:
        class_names = json.load(f)

    # Build model skeleton (no pretrained weights download)
    model = get_model("vit_base_patch16_224", num_classes=26, pretrained=False)

    # Load trained checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Build eval transform from timm model config
    data_config = resolve_model_data_config(model)
    transform = create_transform(**data_config, is_training=False)

    # Extract version info from checkpoint
    epoch = checkpoint.get("epoch", "?")
    val_acc = checkpoint.get("val_acc", None)
    version_parts = ["vit_base_patch16_224", "erasing_adamw", f"epoch={epoch}"]
    if isinstance(val_acc, float):
        version_parts.append(f"val_acc={val_acc:.4f}")
    _state["model_version"] = " | ".join(version_parts)

    _state["model"] = model
    _state["transform"] = transform
    _state["class_names"] = class_names
    _state["device"] = device

    print(f"Model loaded: {_state['model_version']}")
    print(f"Classes: {len(class_names)}")


def is_loaded() -> bool:
    return _state["model"] is not None


def get_model_version() -> str:
    return _state["model_version"] or "not loaded"


def predict(image: Image.Image, top_k: int = 3, threshold: float = 0.5) -> dict:
    """
    Run inference on a PIL Image.
    Returns dict with keys: model_version, top1, topk, warning
    """
    if not is_loaded():
        raise RuntimeError("Model not loaded. Call load_model() first.")

    # Preprocess
    img_tensor = _state["transform"](image.convert("RGB")).unsqueeze(0)
    img_tensor = img_tensor.to(_state["device"])

    # Inference
    with torch.no_grad():
        logits = _state["model"](img_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    # Top-k results
    top_probs, top_indices = torch.topk(probs, k=min(top_k, len(probs)))

    topk = []
    for rank, (prob, idx) in enumerate(
        zip(top_probs.tolist(), top_indices.tolist()), start=1
    ):
        topk.append({"rank": rank, "label": _state["class_names"][idx], "prob": round(prob, 4)})

    top1 = {"label": topk[0]["label"], "prob": topk[0]["prob"]}

    # Warning if confidence below threshold
    warning = None
    if top1["prob"] < threshold:
        warning = (
            f"Low confidence ({top1['prob']:.0%}). "
            "Try a clearer photo with good lighting and minimal blur."
        )

    return {
        "model_version": _state["model_version"],
        "top1": top1,
        "topk": topk,
        "warning": warning,
    }
