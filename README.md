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
- `FACE_MATCH_THRESHOLD`: optional cosine similarity threshold, default `0.6`

If you attach a persistent volume in Coolify, point `FACE_STORAGE_PATH` and `INSIGHTFACE_MODEL_DIR` into that volume so registrations and model downloads survive restarts.
