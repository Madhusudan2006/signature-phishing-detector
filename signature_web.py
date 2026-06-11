"""
cyber_app/backend/signature_app.py
Siamese-network signature verification (FastAPI).
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
import os, tempfile
from typing import Tuple

import numpy as np
import cv2

router = APIRouter()

# ── Model config ──────────────────────────────────────────────────────────────
IMG_SIZE                  = 105
GENUINE_THRESHOLD         = 0.50
DIFFERENT_PERSON_THRESHOLD = 1.00

_BASE       = os.path.dirname(__file__)
MODEL_PATH  = os.path.join(_BASE, "models", "signature_siamese.keras")

_sig_model  = None

def _load_sig():
    global _sig_model
    if _sig_model is not None:
        return _sig_model

    import tensorflow as tf
    from tensorflow.keras import layers

    @tf.keras.utils.register_keras_serializable()
    class DistanceLayer(layers.Layer):
        def call(self, inputs):
            x, y = inputs
            return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True) + 1e-10)

    def euclidean_distance(vectors):
        x, y = vectors
        return tf.sqrt(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True) + 1e-10)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Signature model not found at {MODEL_PATH}. "
            "Please copy signature_siamese.keras into backend/models/."
        )

    _sig_model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"euclidean_distance": euclidean_distance, "DistanceLayer": DistanceLayer},
        compile=False,
        safe_mode=False,
    )
    return _sig_model


# ── Image helpers ─────────────────────────────────────────────────────────────
def _preprocess(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read image file.")
    img = cv2.GaussianBlur(img, (3, 3), 0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y+h, x:x+w]
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=-1)


def _verify(path1: str, path2: str) -> Tuple[str, float, float]:
    model = _load_sig()
    img1  = np.expand_dims(_preprocess(path1), axis=0)
    img2  = np.expand_dims(_preprocess(path2), axis=0)
    dist  = float(model.predict([img1, img2], verbose=0)[0][0])
    sim   = max(0.0, 100.0 - dist * 100.0)

    if dist < GENUINE_THRESHOLD:
        verdict = "GENUINE"
    elif dist < DIFFERENT_PERSON_THRESHOLD:
        verdict = "SUSPICIOUS / FORGED"
    else:
        verdict = "DIFFERENT PERSON"

    return verdict, dist, sim


# ── Routes ────────────────────────────────────────────────────────────────────
@router.post("/verify-signature")
async def verify_signature(
    genuine: UploadFile = File(...),
    suspect: UploadFile = File(...),
):
    tmp1 = tmp2 = None
    try:
        tmp1 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp1.write(await genuine.read()); tmp1.close()
        tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp2.write(await suspect.read()); tmp2.close()

        verdict, dist, sim = _verify(tmp1.name, tmp2.name)

        msg_map = {
            "GENUINE":            "The structural and metric patterns of both signatures align within accepted tolerance thresholds.",
            "SUSPICIOUS / FORGED":"Significant deviations detected. This signature shows characteristics inconsistent with the genuine anchor.",
            "DIFFERENT PERSON":   "The signatures originate from distinctly different writers. Distance exceeds inter-person threshold.",
        }
        return {
            "verdict":    verdict,
            "distance":   dist,
            "similarity": sim,
            "message":    msg_map.get(verdict, ""),
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for t in (tmp1, tmp2):
            if t:
                try: os.unlink(t.name)
                except: pass
