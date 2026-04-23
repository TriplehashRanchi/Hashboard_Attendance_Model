# Facial Recognition API

FastAPI service for face registration and recognition using InsightFace.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Coolify

This repo is set up to deploy as a Python service with Nixpacks:

- `requirements.txt` defines Python dependencies.
- `Procfile` defines the web start command.
- `.python-version` pins a Linux-friendly Python version for deploys.

Recommended environment variables:

- `FACE_API_KEY`: optional API key for `/register` and `/recognize`
- `FACE_STORAGE_PATH`: path to the face embedding store, for example `/data/faces.pkl`
- `INSIGHTFACE_MODEL_DIR`: path for InsightFace model cache, for example `/data/models`
- `INSIGHTFACE_PROVIDERS`: defaults to `CPUExecutionProvider`
- `FACE_MATCH_THRESHOLD`: optional cosine similarity threshold, default `0.62`
- `FACE_MATCH_MARGIN`: optional minimum gap between best and second-best match, default `0.02`
- `FACE_MIN_DETECTION_SCORE`: optional minimum detector confidence, default `0.55`
- `FACE_MIN_FACE_RATIO`: optional minimum face size relative to image, default `0.04`
- `FACE_MIN_SHARPNESS`: optional blur threshold, default `20`
- `FACE_MAX_IMAGE_DIMENSION`: optional max image side before downscaling for performance, default `1280`
- `FACE_REQUIRE_POSE_CHECK`: optional strict landmark geometry check, default `false`

Debug detection without marking attendance:

```bash
curl -F "file=@face.jpg" http://127.0.0.1:8001/inspect
```

If you attach a persistent volume in Coolify, point `FACE_STORAGE_PATH` and `INSIGHTFACE_MODEL_DIR` into that volume so registrations and model downloads survive restarts.
