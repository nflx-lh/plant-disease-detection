"""
Smoke test: hit /health and /predict on the running backend.
Usage: python backend/scripts/smoke_test_predict.py [path/to/image.jpg]
"""

import sys
import json
from pathlib import Path

try:
    import requests
except ImportError:
    print("Install requests first:  pip install requests")
    sys.exit(1)

URL = "http://localhost:8000"


def main():
    # 1. Health check
    print("=== Health Check ===")
    try:
        resp = requests.get(f"{URL}/health", timeout=10)
        print(json.dumps(resp.json(), indent=2))
    except requests.ConnectionError:
        print("ERROR: Cannot connect to backend. Is it running on port 8000?")
        sys.exit(1)

    # 2. Predict
    if len(sys.argv) > 1:
        img_path = Path(sys.argv[1])
    else:
        # Try to find any image in data/ as a fallback
        project_root = Path(__file__).resolve().parents[2]
        candidates = list(project_root.glob("data/**/*.jpg"))[:1] + list(
            project_root.glob("data/**/*.JPG")[:1]
        )
        if not candidates:
            print("\nNo test image found. Pass an image path as argument:")
            print("  python backend/scripts/smoke_test_predict.py path/to/leaf.jpg")
            sys.exit(0)
        img_path = candidates[0]

    print(f"\n=== Predict: {img_path} ===")
    with open(img_path, "rb") as f:
        resp = requests.post(
            f"{URL}/predict",
            files={"file": (img_path.name, f, "image/jpeg")},
            params={"top_k": 3, "threshold": 0.5},
            timeout=30,
        )
    print(f"Status: {resp.status_code}")
    print(json.dumps(resp.json(), indent=2))


if __name__ == "__main__":
    main()
