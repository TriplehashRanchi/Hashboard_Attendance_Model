#!/usr/bin/env python3
"""
Pre-download InsightFace models before the server starts.
Run this in Procfile / release phase so the first HTTP request is never delayed by a download.
"""
import os
import sys
from pathlib import Path

MODEL_DIR = os.getenv("INSIGHTFACE_MODEL_DIR", "/app/models")
MODEL_NAME = os.getenv("INSIGHTFACE_MODEL_NAME", "buffalo_l")
CTX_ID = int(os.getenv("INSIGHTFACE_CTX_ID", "-1"))
DET_SIZE = int(os.getenv("INSIGHTFACE_DET_SIZE", "640"))

Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
print(f"[download_models] model={MODEL_NAME} dir={MODEL_DIR}", flush=True)

try:
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name=MODEL_NAME, root=MODEL_DIR, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=CTX_ID, det_size=(DET_SIZE, DET_SIZE))
    print(f"[download_models] Done — models ready in {MODEL_DIR}", flush=True)
except Exception as exc:
    print(f"[download_models] ERROR: {exc}", flush=True)
    sys.exit(1)
