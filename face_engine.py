import io
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

try:
    from PIL import Image, ImageOps
except Exception:  # pragma: no cover
    Image = None
    ImageOps = None

class FaceEngine:
    def __init__(self, storage_path=None):
        base_dir = os.path.dirname(__file__)
        default_storage_path = os.path.join(base_dir, "data", "faces.pkl")
        self.storage_path = storage_path or os.getenv("FACE_STORAGE_PATH", default_storage_path)
        Path(os.path.dirname(self.storage_path)).mkdir(parents=True, exist_ok=True)

        self.match_threshold = float(os.getenv("FACE_MATCH_THRESHOLD", "0.6"))
        self.min_face_ratio = float(os.getenv("FACE_MIN_FACE_RATIO", "0.07"))
        self.min_sharpness = float(os.getenv("FACE_MIN_SHARPNESS", "55"))
        self.min_brightness = float(os.getenv("FACE_MIN_BRIGHTNESS", "40"))
        self.max_brightness = float(os.getenv("FACE_MAX_BRIGHTNESS", "220"))
        self.max_embeddings_per_user = int(os.getenv("FACE_MAX_EMBEDDINGS_PER_USER", "5"))
        self.require_single_face = os.getenv("FACE_REQUIRE_SINGLE_FACE", "true").lower() == "true"

        ctx_id = int(os.getenv("INSIGHTFACE_CTX_ID", "-1"))
        model_dir = os.getenv("INSIGHTFACE_MODEL_DIR", "/app/models")
        providers = [p.strip() for p in os.getenv("INSIGHTFACE_PROVIDERS", "CPUExecutionProvider").split(",") if p.strip()]

        self.app = FaceAnalysis(
            name=os.getenv("INSIGHTFACE_MODEL_NAME", "buffalo_l"),
            root=model_dir,
            providers=providers,
        )
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        self.known = self._load()  # employeeId -> [embedding1, embedding2, ...]

    def register(self, employee_id, image_bytes):
        img = self._decode_image(image_bytes)
        if img is None:
            return {"success": False, "reason": "invalid_image"}

        faces = self.app.get(img)
        if not faces:
            return {"success": False, "reason": "no_face_found"}
        if self.require_single_face and len(faces) > 1:
            return {"success": False, "reason": "multiple_faces_detected"}

        best_face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        quality = self._check_quality(img, best_face)
        if not quality["ok"]:
            return {
                "success": False,
                "reason": quality["reason"],
                "metrics": quality["metrics"],
            }

        embeddings = self.known.get(employee_id, [])
        embeddings.append(best_face.embedding.astype(np.float32))
        self.known[employee_id] = embeddings[-self.max_embeddings_per_user :]
        self._save()
        return {"success": True, "reason": "registered", "metrics": quality["metrics"]}

    def recognize(self, image_bytes):
        if not self.known:
            return {
                "matched": False,
                "score": None,
                "employeeId": None,
                "reason": "no_registered_faces",
            }

        img = self._decode_image(image_bytes)
        if img is None:
            return {"matched": False, "score": None, "employeeId": None, "reason": "invalid_image"}

        faces = self.app.get(img)
        if not faces:
            return {"matched": False, "score": None, "employeeId": None, "reason": "no_face_found"}
        if self.require_single_face and len(faces) > 1:
            return {
                "matched": False,
                "score": None,
                "employeeId": None,
                "reason": "multiple_faces_detected",
            }

        emb = faces[0].embedding
        quality = self._check_quality(img, faces[0])
        if not quality["ok"]:
            return {
                "matched": False,
                "score": None,
                "employeeId": None,
                "reason": quality["reason"],
                "metrics": quality["metrics"],
            }

        best_id = None
        best_sim = -1.0

        for emp_id, known_emb_list in self.known.items():
            for known_emb in known_emb_list:
                sim = np.dot(emb, known_emb) / (
                    np.linalg.norm(emb) * np.linalg.norm(known_emb)
                )
                if sim > best_sim:
                    best_sim = sim
                    best_id = emp_id

        if best_sim >= self.match_threshold:
            return {
                "matched": True,
                "employeeId": best_id,
                "score": float(best_sim),
                "reason": "matched",
                "metrics": quality["metrics"],
            }

        return {
            "matched": False,
            "employeeId": None,
            "score": float(best_sim),
            "reason": "below_threshold",
            "metrics": quality["metrics"],
        }

    def _load(self):
        if not os.path.exists(self.storage_path):
            return {}
        try:
            with open(self.storage_path, "rb") as handle:
                data = pickle.load(handle)

            normalized = {}
            for key, value in data.items():
                if isinstance(value, list) and value and isinstance(value[0], list):
                    normalized[key] = [np.array(v, dtype=np.float32) for v in value]
                else:
                    normalized[key] = [np.array(value, dtype=np.float32)]
            return normalized
        except Exception:
            return {}

    def _save(self):
        data = {
            key: [embedding.tolist() for embedding in embeddings]
            for key, embeddings in self.known.items()
        }
        with open(self.storage_path, "wb") as handle:
            pickle.dump(data, handle)

    def _decode_image(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            return img

        if Image is None:
            return None

        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            if ImageOps is not None:
                pil_img = ImageOps.exif_transpose(pil_img)
            pil_img = pil_img.convert("RGB")
            arr = np.array(pil_img)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception:
            return None

    def _check_quality(self, img, face):
        h, w = img.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        x1 = max(0, min(x1, w - 1))
        x2 = max(1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(1, min(y2, h))

        face_w = max(1, x2 - x1)
        face_h = max(1, y2 - y1)
        face_ratio = float((face_w * face_h) / max(1, h * w))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())

        metrics = {
            "face_ratio": round(face_ratio, 4),
            "sharpness": round(sharpness, 2),
            "brightness": round(brightness, 2),
        }

        if face_ratio < self.min_face_ratio:
            return {"ok": False, "reason": "face_too_small", "metrics": metrics}
        if sharpness < self.min_sharpness:
            return {"ok": False, "reason": "image_too_blurry", "metrics": metrics}
        if brightness < self.min_brightness:
            return {"ok": False, "reason": "image_too_dark", "metrics": metrics}
        if brightness > self.max_brightness:
            return {"ok": False, "reason": "image_too_bright", "metrics": metrics}

        return {"ok": True, "reason": "ok", "metrics": metrics}
