from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image
import cv2
from deepface import DeepFace

@dataclass
class FaceAnalysisResult:
    embedding: np.ndarray
    face_crop_rgb: np.ndarray
    bbox: Optional[Tuple[int, int, int, int]]
    detector_backend: str
    model_name: str

def pil_to_rgb_array(uploaded_file) -> np.ndarray:
    img = Image.open(uploaded_file).convert("RGB")
    return np.array(img)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)  #compare direction, not size

def similarity_to_score(sim: float) -> float:
    return float(np.clip(sim * 100.0, 0.0, 100.0))

def decide(sim: float, threshold: float) -> str:
    return "OK" if sim >= threshold else "New photo required"

def variance_of_laplacian_blur_score(rgb_img: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def draw_bbox(rgb_img: np.ndarray, bbox: Tuple[int, int, int, int], color=(0, 255, 0), thickness=2) -> np.ndarray:
    x, y, w, h = bbox
    out = rgb_img.copy()
    cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    return out

def manual_face_crop(rgb_img: np.ndarray, bbox: Dict[str, Any]) -> np.ndarray:
    """Extracts the face manually based on the bounding box from the original image."""
    x = int(bbox['x'])
    y = int(bbox['y'])
    w = int(bbox['w'])
    h = int(bbox['h'])
    # Ensure boundaries (won't fail if it goes out of the image)
    img_h, img_w = rgb_img.shape[:2]
    x_end = min(x + w, img_w)
    y_end = min(y + h, img_h)
    x = max(x, 0)
    y = max(y, 0)
    # Return crop (can be empty if bbox is invalid, but this will be checked later)
    return rgb_img[y:y_end, x:x_end]

def _extract_face_and_bbox(
    rgb_img: np.ndarray,
    detector_backend: str,
    enforce_detection: bool,
) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]], Dict[str, Any]]:
    faces = DeepFace.extract_faces(
        img_path=rgb_img,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=True,
    )

    if not faces:
        raise ValueError("No face detected in the image.")

    face0 = faces[0]
    facial_area = face0.get("facial_area")
    bbox = None
    face_crop_manual = None

    if isinstance(facial_area, dict) and all(k in facial_area for k in ("x", "y", "w", "h")):
        bbox = (int(facial_area["x"]), int(facial_area["y"]), int(facial_area["w"]), int(facial_area["h"]))
        face_crop_manual = manual_face_crop(rgb_img, facial_area)
    else:
        raise ValueError("Bounding box for the face not found.")

    if face_crop_manual is None or face_crop_manual.size == 0:
        raise ValueError("Unable to crop the face (empty crop).")

    return face_crop_manual, bbox, face0

def analyze_face(
    rgb_img: np.ndarray,
    model_name: str,
    detector_backend: str = "retinaface",
    enforce_detection: bool = True,
) -> FaceAnalysisResult:
    face_crop, bbox, _raw = _extract_face_and_bbox(
        rgb_img=rgb_img,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
    )

    # Check if the crop is valid (not blank/black)
    if np.std(face_crop) < 15:
        raise ValueError("The cropped face is not valid (appears black) – most likely no face was detected.")

    rep = DeepFace.represent(
        img_path=face_crop,
        model_name=model_name,
        enforce_detection=False,
    )
    embedding = np.asarray(rep[0]["embedding"], dtype=np.float32)

    return FaceAnalysisResult(
        embedding=embedding,
        face_crop_rgb=face_crop,
        bbox=bbox,
        detector_backend=detector_backend,
        model_name=model_name,
    )