import os
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, UploadFile, File, Form
from face_engine import FaceEngine

app = FastAPI()
engine = FaceEngine()
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(5 * 1024 * 1024)))
FACE_API_KEY = os.getenv("FACE_API_KEY")


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


@app.get("/health")
async def health():
    return {"ok": True, "registeredUsers": len(engine.known)}


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

    img = await _read_validated_image(file)
    result = engine.register(emp_id.strip(), img)

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
    img = await _read_validated_image(file)
    result = engine.recognize(img)

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
