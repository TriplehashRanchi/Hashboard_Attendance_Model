import io
import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from insightface.app import FaceAnalysis

try:
    from PIL import Image, ImageOps
except Exception:
    Image = None
    ImageOps = None

# FAISS: fast approximate nearest-neighbor search (optional)
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# boto3: S3 persistence so faces survive redeployment (optional)
try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

logger = logging.getLogger("face_engine")

# ── Data structure ─────────────────────────────────────────────────────────────
# self.known: { employee_id: [ {"emb": np.ndarray, "quality": float}, ... ] }
# Each entry keeps a normalised 512-d ArcFace embedding and the quality score
# from when it was enrolled.  Higher quality → higher weight during matching.


class FaceEngine:
    def __init__(self, storage_path=None):
        base_dir = os.path.dirname(__file__)
        default_data_dir = os.getenv("FACE_DATA_DIR")
        if not default_data_dir:
            default_data_dir = "/data" if os.path.isdir("/data") else os.path.join(base_dir, "data")

        default_storage_path = os.path.join(default_data_dir, "faces.pkl")
        bundled_model_dir = "/app/models" if os.path.isdir("/app/models") else None
        default_model_dir = bundled_model_dir or os.path.join(default_data_dir, "models")

        self.storage_path = storage_path or os.getenv("FACE_STORAGE_PATH", default_storage_path)
        Path(os.path.dirname(self.storage_path)).mkdir(parents=True, exist_ok=True)

        # ── Thresholds ─────────────────────────────────────────────────────────
        self.match_threshold      = float(os.getenv("FACE_MATCH_THRESHOLD",      "0.62"))
        self.match_margin         = float(os.getenv("FACE_MATCH_MARGIN",         "0.02"))
        self.min_detection_score  = float(os.getenv("FACE_MIN_DETECTION_SCORE",  "0.55"))
        self.min_face_ratio       = float(os.getenv("FACE_MIN_FACE_RATIO",       "0.04"))
        self.min_sharpness        = float(os.getenv("FACE_MIN_SHARPNESS",        "20"))
        self.min_brightness       = float(os.getenv("FACE_MIN_BRIGHTNESS",       "40"))
        self.max_brightness       = float(os.getenv("FACE_MAX_BRIGHTNESS",       "220"))
        self.max_image_dimension  = int(os.getenv("FACE_MAX_IMAGE_DIMENSION",    "1280"))
        self.max_embeddings_per_user = int(os.getenv("FACE_MAX_EMBEDDINGS_PER_USER", "5"))
        self.require_single_face  = os.getenv("FACE_REQUIRE_SINGLE_FACE",  "true").lower()  == "true"
        self.require_pose_check   = os.getenv("FACE_REQUIRE_POSE_CHECK",   "false").lower() == "true"
        self.replace_on_register  = os.getenv("FACE_REPLACE_ON_REGISTER",  "true").lower()  == "true"

        # ── Anti-spoofing ──────────────────────────────────────────────────────
        # FACE_ANTISPOOF_ENABLED=true activates liveness check.
        # Uses InsightFace's minivision model; soft-fails if not available.
        self.antispoof_enabled = os.getenv("FACE_ANTISPOOF_ENABLED", "false").lower() == "true"
        self.antispoof_threshold = float(os.getenv("FACE_ANTISPOOF_THRESHOLD", "0.5"))
        self._antispoof_model = None
        if self.antispoof_enabled:
            self._antispoof_model = self._load_antispoof_model()

        # ── InsightFace model ──────────────────────────────────────────────────
        ctx_id = int(os.getenv("INSIGHTFACE_CTX_ID", "-1"))
        det_size_value = int(os.getenv("INSIGHTFACE_DET_SIZE", "640"))
        model_dir = os.getenv("INSIGHTFACE_MODEL_DIR") or default_model_dir
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        providers = [
            p.strip()
            for p in os.getenv("INSIGHTFACE_PROVIDERS", "CPUExecutionProvider").split(",")
            if p.strip()
        ]

        logger.info(
            "Initializing FaceEngine storage=%s model_dir=%s providers=%s "
            "threshold=%.2f antispoof=%s faiss=%s",
            self.storage_path, model_dir, providers,
            self.match_threshold, self.antispoof_enabled, FAISS_AVAILABLE,
        )

        self.app = FaceAnalysis(
            name=os.getenv("INSIGHTFACE_MODEL_NAME", "buffalo_l"),
            root=model_dir,
            providers=providers,
        )
        self.app.prepare(ctx_id=ctx_id, det_size=(det_size_value, det_size_value))

        # ── S3 sync ────────────────────────────────────────────────────────────
        self._s3_bucket = os.getenv("S3_BUCKET")
        self._s3_key    = os.getenv("S3_KEY", "faces.pkl")
        if self._s3_bucket and BOTO3_AVAILABLE:
            self._s3_client = boto3.client(
                "s3",
                region_name=os.getenv("AWS_REGION", "us-east-1"),
            )
            self._s3_download()
        else:
            self._s3_client = None

        self.known = self._load()
        self._faiss_index = None
        self._faiss_map: list[str] = []   # index position → employee_id
        self._rebuild_faiss()

        self.faiss_enabled = FAISS_AVAILABLE and self._faiss_index is not None
        logger.info("FaceEngine ready registered_users=%s", len(self.known))

    # ── Public API ─────────────────────────────────────────────────────────────

    def register(self, employee_id: str, image_bytes: bytes) -> dict:
        return self.register_batch(employee_id, [image_bytes], replace=self.replace_on_register)

    def register_batch(
        self,
        employee_id: str,
        images: list[bytes],
        replace: Optional[bool] = None,
    ) -> dict:
        """Register one or more images for an employee.

        Accepts up to 5 images; filters out low-quality frames; keeps the
        best-quality embeddings (up to FACE_MAX_EMBEDDINGS_PER_USER).
        """
        if replace is None:
            replace = self.replace_on_register

        accepted: list[dict] = []
        rejected: list[str]  = []

        for idx, image_bytes in enumerate(images):
            img = self._decode_image(image_bytes)
            if img is None:
                rejected.append(f"image_{idx}:invalid_image")
                continue

            img = self._prepare_image(img)
            faces = self.app.get(img)
            if not faces:
                rejected.append(f"image_{idx}:no_face_found")
                continue
            if self.require_single_face and len(faces) > 1:
                rejected.append(f"image_{idx}:multiple_faces_detected")
                continue

            best_face = self._select_best_face(faces)
            quality   = self._check_quality(img, best_face)
            if not quality["ok"]:
                rejected.append(f"image_{idx}:{quality['reason']}")
                continue

            if self.antispoof_enabled and self._antispoof_model is not None:
                real, spoof_score = self._check_liveness(img, best_face)
                if not real:
                    rejected.append(f"image_{idx}:liveness_failed(score={spoof_score:.2f})")
                    continue

            quality_score = self._compute_quality_score(quality["metrics"])
            accepted.append({
                "emb":     self._normalize_embedding(best_face.embedding),
                "quality": quality_score,
                "metrics": quality["metrics"],
            })

        if not accepted:
            logger.info(
                "register_batch rejected employee_id=%s rejected=%s",
                employee_id, rejected,
            )
            return {
                "success": False,
                "reason":  "all_images_rejected",
                "rejected": rejected,
                "accepted": 0,
                "total":    len(images),
            }

        # Sort by quality descending and keep the cap
        accepted.sort(key=lambda x: x["quality"], reverse=True)
        existing = [] if replace else self.known.get(employee_id, [])
        merged   = accepted + existing
        merged.sort(key=lambda x: x["quality"], reverse=True)
        self.known[employee_id] = merged[: self.max_embeddings_per_user]

        self._save()
        self._rebuild_faiss()

        logger.info(
            "register_batch success employee_id=%s accepted=%s total=%s stored=%s rejected=%s",
            employee_id, len(accepted), len(images),
            len(self.known[employee_id]), rejected,
        )
        return {
            "success":  True,
            "reason":   "registered",
            "accepted": len(accepted),
            "total":    len(images),
            "stored":   len(self.known[employee_id]),
            "rejected": rejected,
            "metrics":  accepted[0]["metrics"],
        }

    def recognize(self, image_bytes: bytes) -> dict:
        if not self.known:
            return {"matched": False, "score": None, "employeeId": None, "reason": "no_registered_faces"}

        img = self._decode_image(image_bytes)
        if img is None:
            return {"matched": False, "score": None, "employeeId": None, "reason": "invalid_image"}

        img = self._prepare_image(img)
        faces = self.app.get(img)
        if not faces:
            return {"matched": False, "score": None, "employeeId": None, "reason": "no_face_found"}
        if self.require_single_face and len(faces) > 1:
            return {"matched": False, "score": None, "employeeId": None, "reason": "multiple_faces_detected"}

        best_face = self._select_best_face(faces)
        quality   = self._check_quality(img, best_face)
        if not quality["ok"]:
            return {
                "matched":    False,
                "score":      None,
                "employeeId": None,
                "reason":     quality["reason"],
                "metrics":    quality["metrics"],
            }

        if self.antispoof_enabled and self._antispoof_model is not None:
            real, spoof_score = self._check_liveness(img, best_face)
            if not real:
                logger.info("recognize rejected reason=liveness_failed score=%.2f", spoof_score)
                return {
                    "matched":    False,
                    "score":      None,
                    "employeeId": None,
                    "reason":     "liveness_failed",
                    "metrics":    quality["metrics"],
                }

        emb = self._normalize_embedding(best_face.embedding)
        best_id, best_sim, second_sim = self._find_best_match(emb)

        if best_sim >= self.match_threshold and (best_sim - second_sim) >= self.match_margin:
            logger.info(
                "recognize matched employee_id=%s score=%.4f margin=%.4f",
                best_id, best_sim, best_sim - second_sim,
            )
            return {
                "matched":    True,
                "employeeId": best_id,
                "score":      float(best_sim),
                "reason":     "matched",
                "metrics":    quality["metrics"],
            }

        reason = "ambiguous_match" if best_sim >= self.match_threshold else "below_threshold"
        logger.info(
            "recognize rejected reason=%s best_id=%s score=%.4f second=%.4f",
            reason, best_id, best_sim, second_sim,
        )
        return {
            "matched":    False,
            "employeeId": None,
            "score":      float(best_sim),
            "reason":     reason,
            "metrics":    quality["metrics"],
        }

    def inspect(self, image_bytes: bytes) -> dict:
        img = self._decode_image(image_bytes)
        if img is None:
            return {"ok": False, "reason": "invalid_image", "faces": []}

        img = self._prepare_image(img)
        faces = self.app.get(img)
        if not faces:
            return {"ok": False, "reason": "no_face_found", "faces": []}

        inspected = []
        for face in faces:
            q = self._check_quality(img, face)
            inspected.append({"ok": q["ok"], "reason": q["reason"], "metrics": q["metrics"]})

        best = self._select_best_face(faces)
        bq   = self._check_quality(img, best)
        return {
            "ok":        bq["ok"],
            "reason":    bq["reason"],
            "faceCount": len(faces),
            "faces":     inspected,
            "bestFace":  bq,
        }

    # ── Matching ───────────────────────────────────────────────────────────────

    def _find_best_match(self, emb: np.ndarray) -> tuple[Optional[str], float, float]:
        """Return (best_employee_id, best_similarity, second_best_similarity).

        Uses FAISS when available (fast for large N), otherwise numpy linear scan.
        Scoring is quality-weighted: embeddings enrolled from better-quality photos
        contribute more to the aggregated per-employee score.
        """
        if FAISS_AVAILABLE and self._faiss_index is not None and self._faiss_index.ntotal > 0:
            return self._find_best_match_faiss(emb)
        return self._find_best_match_numpy(emb)

    def _find_best_match_faiss(self, emb: np.ndarray) -> tuple[Optional[str], float, float]:
        query = emb.reshape(1, -1).astype(np.float32)
        k     = min(self._faiss_index.ntotal, max(10, len(self.known) * 2))
        sims, idxs = self._faiss_index.search(query, k)

        # Aggregate per-employee: quality-weighted max
        scores: dict[str, float] = {}
        weights: dict[str, float] = {}
        for sim, idx in zip(sims[0], idxs[0]):
            if idx < 0:
                continue
            emp_id = self._faiss_map[idx]
            entry  = self.known[emp_id]
            # find quality for this particular embedding by index within employee
            emp_emb_idx = self._faiss_map[:idx + 1].count(emp_id) - 1
            q = entry[emp_emb_idx]["quality"] if emp_emb_idx < len(entry) else 1.0
            weighted = float(sim) * q
            if emp_id not in scores or weighted > scores[emp_id]:
                scores[emp_id] = weighted
                weights[emp_id] = q

        if not scores:
            return None, -1.0, -1.0

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
        best_id    = sorted_ids[0]
        # Normalize back to cosine range for threshold comparison
        best_sim   = scores[best_id] / max(weights[best_id], 1e-6)
        second_sim = (
            scores[sorted_ids[1]] / max(weights[sorted_ids[1]], 1e-6)
            if len(sorted_ids) > 1 else -1.0
        )
        return best_id, best_sim, second_sim

    def _find_best_match_numpy(self, emb: np.ndarray) -> tuple[Optional[str], float, float]:
        best_id, best_sim, second_sim = None, -1.0, -1.0
        for emp_id, entries in self.known.items():
            emp_best = -1.0
            for entry in entries:
                sim = float(np.dot(emb, entry["emb"])) * entry["quality"]
                if sim > emp_best:
                    emp_best = sim
            # Normalize back
            max_q = max(e["quality"] for e in entries)
            emp_score = emp_best / max(max_q, 1e-6)
            if emp_score > best_sim:
                second_sim = best_sim
                best_sim   = emp_score
                best_id    = emp_id
            elif emp_score > second_sim:
                second_sim = emp_score

        return best_id, best_sim, second_sim

    # ── FAISS index ────────────────────────────────────────────────────────────

    def _rebuild_faiss(self):
        if not FAISS_AVAILABLE:
            return

        all_embs: list[np.ndarray] = []
        emp_map:  list[str]        = []

        for emp_id, entries in self.known.items():
            for entry in entries:
                all_embs.append(entry["emb"])
                emp_map.append(emp_id)

        dim = 512
        index = faiss.IndexFlatIP(dim)   # inner product ≡ cosine on normalised vecs
        if all_embs:
            matrix = np.stack(all_embs).astype(np.float32)
            faiss.normalize_L2(matrix)    # ensure unit length
            index.add(matrix)

        self._faiss_index = index
        self._faiss_map   = emp_map

    # ── Anti-spoofing ──────────────────────────────────────────────────────────

    def _load_antispoof_model(self):
        try:
            from insightface.model_zoo import get_model
            model_dir = os.getenv("INSIGHTFACE_MODEL_DIR", "/app/models")
            providers  = [
                p.strip()
                for p in os.getenv("INSIGHTFACE_PROVIDERS", "CPUExecutionProvider").split(",")
                if p.strip()
            ]
            model = get_model("minivision", root=model_dir, providers=providers)
            model.prepare(ctx_id=int(os.getenv("INSIGHTFACE_CTX_ID", "-1")), input_size=(80, 80))
            logger.info("Anti-spoofing model loaded (minivision)")
            return model
        except Exception as exc:
            logger.warning("Anti-spoofing model not available: %s — checks disabled", exc)
            return None

    def _check_liveness(self, img: np.ndarray, face) -> tuple[bool, float]:
        try:
            is_real, score = self._antispoof_model.predict(img, face.bbox)
            return bool(is_real), float(score)
        except Exception as exc:
            logger.warning("Anti-spoof check failed: %s — treating as real", exc)
            return True, 1.0

    # ── Storage ────────────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if not os.path.exists(self.storage_path):
            return {}
        try:
            with open(self.storage_path, "rb") as fh:
                raw = pickle.load(fh)

            normalized: dict = {}
            for emp_id, value in raw.items():
                # Migrate old format (list of raw embeddings) to new format
                if isinstance(value, list) and value and not isinstance(value[0], dict):
                    entries = []
                    for v in value:
                        arr = np.array(v, dtype=np.float32)
                        entries.append({
                            "emb":     self._normalize_embedding(arr),
                            "quality": 1.0,    # unknown quality → neutral weight
                        })
                    normalized[emp_id] = entries
                else:
                    entries = []
                    for entry in value:
                        emb = np.array(entry["emb"], dtype=np.float32)
                        entries.append({
                            "emb":     self._normalize_embedding(emb),
                            "quality": float(entry.get("quality", 1.0)),
                        })
                    normalized[emp_id] = entries

            logger.info("Loaded face storage: %s users", len(normalized))
            return normalized
        except Exception as exc:
            logger.exception("Failed to load face storage: %s", exc)
            return {}

    def _save(self):
        data = {
            emp_id: [
                {"emb": e["emb"].tolist(), "quality": e["quality"]}
                for e in entries
            ]
            for emp_id, entries in self.known.items()
        }
        with open(self.storage_path, "wb") as fh:
            pickle.dump(data, fh)

        if self._s3_client and self._s3_bucket:
            self._s3_upload()

    # ── S3 helpers ─────────────────────────────────────────────────────────────

    def _s3_download(self):
        """Pull faces.pkl from S3 if the local file is absent or older."""
        try:
            self._s3_client.download_file(self._s3_bucket, self._s3_key, self.storage_path)
            logger.info("Downloaded faces.pkl from s3://%s/%s", self._s3_bucket, self._s3_key)
        except Exception as exc:
            # File doesn't exist on S3 yet — that's fine on first deploy
            logger.info("S3 download skipped: %s", exc)

    def _s3_upload(self):
        try:
            self._s3_client.upload_file(self.storage_path, self._s3_bucket, self._s3_key)
            logger.info("Uploaded faces.pkl to s3://%s/%s", self._s3_bucket, self._s3_key)
        except Exception as exc:
            logger.error("S3 upload failed: %s", exc)

    # ── Image helpers ──────────────────────────────────────────────────────────

    def _decode_image(self, image_bytes: bytes) -> Optional[np.ndarray]:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None:
            return img

        if Image is None:
            return None

        try:
            pil = Image.open(io.BytesIO(image_bytes))
            if ImageOps is not None:
                pil = ImageOps.exif_transpose(pil)
            pil = pil.convert("RGB")
            return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception:
            return None

    def _prepare_image(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        largest = max(h, w)
        if largest <= self.max_image_dimension:
            return img
        scale    = self.max_image_dimension / largest
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    def _select_best_face(self, faces):
        return max(
            faces,
            key=lambda f: (
                (f.bbox[2] - f.bbox[0])
                * (f.bbox[3] - f.bbox[1])
                * max(0.01, float(getattr(f, "det_score", 1.0)))
            ),
        )

    # ── Quality ────────────────────────────────────────────────────────────────

    def _check_quality(self, img: np.ndarray, face) -> dict:
        h, w    = img.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        x1 = max(0, min(x1, w - 1));  x2 = max(1, min(x2, w))
        y1 = max(0, min(y1, h - 1));  y2 = max(1, min(y2, h))

        face_w     = max(1, x2 - x1)
        face_h     = max(1, y2 - y1)
        face_ratio = float((face_w * face_h) / max(1, h * w))

        gray       = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpness  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        det_score  = float(getattr(face, "det_score", 1.0))

        metrics = {
            "face_ratio":       round(face_ratio, 4),
            "sharpness":        round(sharpness,  2),
            "brightness":       round(brightness, 2),
            "detection_score":  round(det_score,  4),
        }

        if det_score  < self.min_detection_score: return {"ok": False, "reason": "low_detection_confidence", "metrics": metrics}
        if face_ratio < self.min_face_ratio:      return {"ok": False, "reason": "face_too_small",           "metrics": metrics}
        if sharpness  < self.min_sharpness:       return {"ok": False, "reason": "image_too_blurry",         "metrics": metrics}
        if brightness < self.min_brightness:      return {"ok": False, "reason": "image_too_dark",           "metrics": metrics}
        if brightness > self.max_brightness:      return {"ok": False, "reason": "image_too_bright",         "metrics": metrics}

        if self.require_pose_check:
            ok, reason = self._check_pose(face, face_w, face_h)
            if not ok:
                return {"ok": False, "reason": reason, "metrics": metrics}

        return {"ok": True, "reason": "ok", "metrics": metrics}

    def _compute_quality_score(self, metrics: dict) -> float:
        """Map quality metrics to a [0.5, 1.0] weight for embedding storage.

        Higher score → this enrolled photo was clearer → trusted more during match.
        """
        det    = min(metrics.get("detection_score", 1.0), 1.0)
        sharp  = min(metrics.get("sharpness", 100) / 200.0, 1.0)
        bright = metrics.get("brightness", 128)
        bright_score = 1.0 - abs(bright - 128) / 128.0   # peaks at 128, drops at extremes
        ratio  = min(metrics.get("face_ratio", 0.1) / 0.25, 1.0)

        score = 0.35 * det + 0.30 * sharp + 0.20 * bright_score + 0.15 * ratio
        return max(0.5, round(score, 4))

    def _check_pose(self, face, face_w: int, face_h: int) -> tuple[bool, str]:
        kps = getattr(face, "kps", None)
        if kps is None or len(kps) < 5:
            return False, "landmarks_missing"

        left_eye, right_eye, nose, left_mouth, right_mouth = [
            np.array(pt, dtype=np.float32) for pt in kps[:5]
        ]
        eye_mid      = (left_eye + right_eye) / 2.0
        mouth_mid    = (left_mouth + right_mouth) / 2.0
        eye_dist     = float(np.linalg.norm(right_eye - left_eye))
        mouth_dist   = float(np.linalg.norm(right_mouth - left_mouth))
        eye_slope    = abs(float(right_eye[1] - left_eye[1])) / max(1.0, face_h)
        nose_offset  = abs(float(nose[0] - eye_mid[0])) / max(1.0, face_w)

        if eye_dist / max(1.0, face_w)               < 0.22: return False, "face_not_close_enough"
        if mouth_dist / max(1.0, face_w)              < 0.15: return False, "lower_face_occluded"
        if eye_slope                                  > 0.08: return False, "face_tilted"
        if float(nose[1] - eye_mid[1]) / max(1.0, face_h) < 0.10: return False, "pose_not_frontal"
        if float(mouth_mid[1] - nose[1]) / max(1.0, face_h) < 0.10: return False, "lower_face_occluded"
        if nose_offset                                > 0.12: return False, "face_not_centered"

        return True, "ok"

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        emb  = embedding.astype(np.float32)
        norm = np.linalg.norm(emb)
        return emb if norm <= 0 else emb / norm
