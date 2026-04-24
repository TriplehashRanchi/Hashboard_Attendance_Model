import os
import logging
import time
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request, UploadFile, File, Form
from face_engine import FaceEngine

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("face_api")

app = FastAPI()
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(5 * 1024 * 1024)))
FACE_API_KEY = os.getenv("FACE_API_KEY")
engine: Optional[FaceEngine] = None
engine_init_error: Optional[str] = None


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
    global engine, engine_init_error
    if engine is not None:
        return engine

    try:
        logger.info("Initializing FaceEngine")
        engine = FaceEngine()
        engine_init_error = None
        return engine
    except Exception as exc:
        engine_init_error = str(exc)
        logger.exception("FaceEngine initialization failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Face engine initialization failed: {engine_init_error}",
        ) from exc


@app.get("/health")
async def health():
    if engine is None and engine_init_error is None:
        try:
            get_engine()
        except HTTPException:
            pass

    if engine_init_error:
        return {
            "ok": False,
            "error": engine_init_error,
        }

    if engine is None:
        return {
            "ok": False,
            "error": "Face engine not initialized",
        }

    return {
        "ok": True,
        "registeredUsers": len(engine.known),
        "storagePath": engine.storage_path,
        "knownUsers": list(engine.known.keys()),
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
    logger.info("inspect_result ok=%s reason=%s face_count=%s", result.get("ok"), result.get("reason"), result.get("faceCount"))
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )
