"""
cyber_app/backend/signature_app.py
Siamese-network signature verification + OpenCV handwriting analysis.
Adapted from sig_app/main.py (Streamlit → FastAPI).
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
import os, tempfile, math, random
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

    # Only import tensorflow when actually needed
    import tensorflow as tf
    from tensorflow.keras import layers

    @tf.keras.utils.register_keras_serializable()
    class DistanceLayer(layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
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


# ── Handwriting analysis (OpenCV-based graphology) ───────────────────────────
def _analyze_handwriting(path: str) -> dict:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read image file.")

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ── Slant estimation via Hough lines ──────────────────────────────────────
    edges  = cv2.Canny(binary, 50, 150)
    lines  = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30,
                              minLineLength=20, maxLineGap=5)
    angles = []
    if lines is not None:
        for ln in lines:
            x1,y1,x2,y2 = ln[0]
            if x2 != x1:
                angles.append(math.degrees(math.atan2(y2-y1, x2-x1)))
    avg_angle = float(np.mean(angles)) if angles else 0.0
    if   avg_angle < -5:  slant = "Right-leaning"
    elif avg_angle >  5:  slant = "Left-leaning"
    else:                  slant = "Upright / Vertical"

    # ── Baseline trend ────────────────────────────────────────────────────────
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids_y = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] > 0:
            centroids_y.append(M["m01"] / M["m00"])
    if len(centroids_y) > 2:
        xs  = np.arange(len(centroids_y))
        fit = np.polyfit(xs, centroids_y, 1)
        if   fit[0] < -0.3: trend = "Ascending"
        elif fit[0] >  0.3: trend = "Descending"
        else:                 trend = "Stable / Straight"
    else:
        trend = "Indeterminate"

    # ── Pen pressure (mean intensity of ink pixels) ───────────────────────────
    ink_pixels = img[binary > 0]
    if len(ink_pixels):
        mean_val  = float(np.mean(255 - ink_pixels))
        pressure  = "Heavy" if mean_val > 160 else ("Medium" if mean_val > 100 else "Light")
    else:
        pressure = "Unknown"

    # ── Word spacing (gap histogram) ──────────────────────────────────────────
    proj   = np.sum(binary, axis=0)
    gaps   = np.where(proj == 0)[0]
    if len(gaps) > 1:
        gap_sz = float(np.mean(np.diff(gaps)))
        spacing = "Wide" if gap_sz > 15 else ("Normal" if gap_sz > 6 else "Narrow")
    else:
        spacing = "Normal"

    # ── Baseline alignment ────────────────────────────────────────────────────
    row_proj = np.sum(binary, axis=1)
    baseline = "Consistent" if float(np.std(row_proj)) < 500 else "Irregular"

    # ── Predicted gender (graphology heuristic — illustrative only) ──────────
    gender_score = 0
    if slant   == "Right-leaning": gender_score += 1
    if pressure == "Light":        gender_score += 1
    if spacing  == "Wide":         gender_score += 1
    gender = "Female (tendency)" if gender_score >= 2 else "Male (tendency)"

    # ── Personality trait pills ───────────────────────────────────────────────
    traits = []
    if slant   == "Right-leaning":   traits.append("Expressive & Sociable")
    elif slant == "Left-leaning":    traits.append("Introverted / Reserved")
    else:                             traits.append("Balanced & Controlled")
    if pressure == "Heavy":          traits.append("Intense Focus")
    elif pressure == "Light":        traits.append("Sensitive & Perceptive")
    if trend   == "Ascending":       traits.append("Optimistic Outlook")
    elif trend == "Descending":      traits.append("Cautious / Analytical")
    if spacing == "Wide":            traits.append("Open-minded & Creative")
    elif spacing == "Narrow":        traits.append("Detail-oriented & Precise")
    if baseline == "Consistent":     traits.append("Disciplined & Structured")
    else:                             traits.append("Flexible & Adaptive")

    return {
        "metrics": {
            "slant":    slant,
            "baseline": baseline,
            "pressure": pressure,
            "spacing":  spacing,
            "trend":    trend,
            "gender":   gender,
        },
        "traits": traits,
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@router.post("/analyze-handwriting")
async def analyze_handwriting(sample: UploadFile = File(...)):
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp.write(await sample.read())
        tmp.close()
        result = _analyze_handwriting(tmp.name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try: os.unlink(tmp.name)
        except: pass


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
