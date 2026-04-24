import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request, UploadFile, File, Form
from face_engine import FaceEngine

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("face_api")

MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(5 * 1024 * 1024)))
FACE_API_KEY = os.getenv("FACE_API_KEY")
engine: Optional[FaceEngine] = None
engine_init_error: Optional[str] = None


# ── Startup: eagerly load model so first request is never delayed ──────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, engine_init_error
    loop = asyncio.get_event_loop()
    try:
        logger.info("Starting FaceEngine initialization (this may take a moment)...")
        engine = await loop.run_in_executor(None, FaceEngine)
        logger.info("FaceEngine ready — registered_users=%s", len(engine.known))
    except Exception as exc:
        engine_init_error = str(exc)
        logger.exception("FaceEngine startup failed: %s", exc)
    yield
    engine = None


app = FastAPI(lifespan=lifespan)


# ── Middleware ─────────────────────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "request method=%s path=%s status=%s elapsed_ms=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response
    except Exception:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.exception(
            "request_failed method=%s path=%s elapsed_ms=%s",
            request.method,
            request.url.path,
            elapsed_ms,
        )
        raise


# ── Helpers ────────────────────────────────────────────────────────────────────

def _validate_api_key(x_api_key: Optional[str]) -> None:
    if FACE_API_KEY and x_api_key != FACE_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


async def _read_validated_image(file: UploadFile) -> bytes:
    if not file:
        raise HTTPException(status_code=400, detail="No image received")
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Only image uploads are allowed")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image received")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds max size of {MAX_IMAGE_BYTES} bytes",
        )
    return data


def get_engine() -> FaceEngine:
    if engine is not None:
        return engine
    if engine_init_error:
        raise HTTPException(
            status_code=503,
            detail=f"Face engine failed to start: {engine_init_error}",
        )
    raise HTTPException(status_code=503, detail="Face engine not ready yet")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    if engine_init_error:
        return {"ok": False, "error": engine_init_error}
    if engine is None:
        return {"ok": False, "error": "Face engine not initialized"}
    return {
        "ok": True,
        "registeredUsers": len(engine.known),
        "storagePath": engine.storage_path,
        "knownUsers": list(engine.known.keys()),
        "antispoof": engine.antispoof_enabled,
        "faiss": engine.faiss_enabled,
    }


@app.post("/register")
async def register(
    employeeId: Optional[str] = Form(None),
    userId: Optional[str] = Form(None),
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(None),
):
    _validate_api_key(x_api_key)
    emp_id = employeeId or userId
    if not emp_id:
        raise HTTPException(status_code=400, detail="Missing employeeId")

    current_engine = get_engine()
    img = await _read_validated_image(file)
    result = current_engine.register(emp_id.strip(), img)
    logger.info(
        "register_result employee_id=%s success=%s reason=%s metrics=%s",
        emp_id.strip(),
        result.get("success"),
        result.get("reason"),
        result.get("metrics"),
    )

    if result["success"]:
        return {
            "success": True,
            "message": "Registered",
            "reason": result.get("reason"),
            "metrics": result.get("metrics"),
        }
    return {
        "success": False,
        "message": "Registration failed",
        "reason": result.get("reason"),
        "metrics": result.get("metrics"),
    }


@app.post("/register-batch")
async def register_batch(
    employeeId: Optional[str] = Form(None),
    userId: Optional[str] = Form(None),
    files: list[UploadFile] = File(...),
    x_api_key: Optional[str] = Header(None),
):
    """Register multiple photos at once — stores the best-quality embeddings."""
    _validate_api_key(x_api_key)
    emp_id = (employeeId or userId or "").strip()
    if not emp_id:
        raise HTTPException(status_code=400, detail="Missing employeeId")
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    current_engine = get_engine()
    images = []
    for f in files:
        data = await _read_validated_image(f)
        images.append(data)

    result = current_engine.register_batch(emp_id, images)
    logger.info(
        "register_batch_result employee_id=%s success=%s accepted=%s total=%s",
        emp_id,
        result.get("success"),
        result.get("accepted"),
        result.get("total"),
    )
    return result


@app.post("/recognize")
async def recognize(
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(None),
):
    _validate_api_key(x_api_key)
    current_engine = get_engine()
    img = await _read_validated_image(file)
    result = current_engine.recognize(img)
    logger.info(
        "recognize_result matched=%s employee_id=%s score=%s reason=%s metrics=%s",
        result.get("matched"),
        result.get("employeeId"),
        result.get("score"),
        result.get("reason"),
        result.get("metrics"),
    )

    if result["matched"]:
        return {
            "matched": True,
            "employeeId": result["employeeId"],
            "score": result["score"],
            "reason": result.get("reason"),
            "metrics": result.get("metrics"),
        }
    return {
        "matched": False,
        "score": result["score"],
        "reason": result.get("reason"),
        "metrics": result.get("metrics"),
    }


@app.post("/inspect")
async def inspect(
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(None),
):
    _validate_api_key(x_api_key)
    current_engine = get_engine()
    img = await _read_validated_image(file)
    result = current_engine.inspect(img)
    logger.info(
        "inspect_result ok=%s reason=%s face_count=%s",
        result.get("ok"),
        result.get("reason"),
        result.get("faceCount"),
    )
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )
