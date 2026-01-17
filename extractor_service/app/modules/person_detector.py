"""
Person detection utilities for face detection, blur scoring, and pose categorization.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from ..perf import profile_time

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - optional dependency
    YOLO = None

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

_YOLO_MODEL = None
_YOLO_DEVICE = None
_HAAR_CASCADE = None
_WARNED_NO_YOLO = False
_WARNED_YOLO_FAILED = False
_WARNED_NO_CASCADE = False

_YOLO_MODEL_NAME = os.getenv("PERFECTFRAMEAI_YOLO_MODEL", "yolov8n-face.pt")
_YOLO_CONFIDENCE = float(os.getenv("PERFECTFRAMEAI_YOLO_CONFIDENCE", "0.25"))
_YOLO_IMAGE_SIZE = int(os.getenv("PERFECTFRAMEAI_YOLO_IMAGE_SIZE", "640"))

# Normalization constants for scoring.
_FACE_SIZE_NORMALIZATION = 0.2
_CLARITY_NORMALIZATION = 200.0
_PORTRAIT_THRESHOLD = 0.15
_FULL_BODY_THRESHOLD = 0.05
_PROFILE_ASPECT_RATIO_THRESHOLD = 0.75


@dataclass(frozen=True)
class FaceDetection:
    """Detected face bounding box and confidence."""
    bbox: Tuple[int, int, int, int]
    confidence: float

    @property
    def area(self) -> int:
        """Return face area in pixels."""
        return max(self.bbox[2], 0) * max(self.bbox[3], 0)


def _get_yolo_device() -> str:
    """Select YOLO inference device with GPU support when available."""
    global _YOLO_DEVICE
    if _YOLO_DEVICE is not None:
        return _YOLO_DEVICE
    override = os.getenv("PERFECTFRAMEAI_YOLO_DEVICE")
    if override:
        _YOLO_DEVICE = override
        logger.info("YOLO face detector device override: %s", _YOLO_DEVICE)
        return _YOLO_DEVICE
    if torch is not None and torch.cuda.is_available():
        _YOLO_DEVICE = "cuda:0"
    else:
        _YOLO_DEVICE = "cpu"
    logger.info("YOLO face detector using device: %s", _YOLO_DEVICE)
    return _YOLO_DEVICE


def _get_yolo_model():
    """Return a cached YOLO face detector instance, if available."""
    global _YOLO_MODEL
    if YOLO is None:
        return None
    if os.getenv("PERFECTFRAMEAI_FORCE_HAAR"):
        return None
    if _YOLO_MODEL is None:
        try:
            _YOLO_MODEL = YOLO(_YOLO_MODEL_NAME)
        except Exception as exc:  # pragma: no cover - defensive path
            global _WARNED_YOLO_FAILED
            if not _WARNED_YOLO_FAILED:
                logger.warning("YOLO model load failed: %s", exc)
                _WARNED_YOLO_FAILED = True
            return None
    return _YOLO_MODEL


def _get_haar_cascade():
    """Return a cached OpenCV Haar cascade for face detection."""
    global _HAAR_CASCADE
    if _HAAR_CASCADE is not None:
        return _HAAR_CASCADE
    cascade_root = getattr(cv2.data, "haarcascades", "")
    if not cascade_root:
        return None
    cascade_path = os.path.join(cascade_root, "haarcascade_frontalface_default.xml")
    if not os.path.exists(cascade_path):
        return None
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        return None
    _HAAR_CASCADE = cascade
    return _HAAR_CASCADE


def _detect_faces_with_haar(frame: np.ndarray) -> List[FaceDetection]:
    """Fallback face detection using OpenCV Haar cascades."""
    global _WARNED_NO_CASCADE
    cascade = _get_haar_cascade()
    if cascade is None:
        if not _WARNED_NO_CASCADE:
            logger.warning("OpenCV Haar cascade not available. Face detection disabled.")
            _WARNED_NO_CASCADE = True
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return [FaceDetection(bbox=(x, y, w, h), confidence=1.0) for (x, y, w, h) in faces]


@profile_time("detect_faces")
def detect_faces(frame: np.ndarray) -> List[FaceDetection]:
    """Detect faces and return bounding boxes with confidence scores."""
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        return []

    detector = _get_yolo_model()
    if detector is None:
        global _WARNED_NO_YOLO
        if not _WARNED_NO_YOLO:
            if YOLO is None:
                logger.warning("YOLO not available. Falling back to OpenCV Haar cascade.")
            else:
                logger.warning("YOLO unavailable. Falling back to OpenCV Haar cascade.")
            _WARNED_NO_YOLO = True
        return _detect_faces_with_haar(frame)

    try:
        device = _get_yolo_device()
        results = detector.predict(
            frame,
            imgsz=_YOLO_IMAGE_SIZE,
            conf=_YOLO_CONFIDENCE,
            device=device,
            verbose=False,
            half=(device != "cpu"),
        )
    except Exception as exc:  # pragma: no cover - defensive path
        logger.warning("YOLO face detection failed: %s", exc)
        return _detect_faces_with_haar(frame)

    if not results:
        return []

    result = results[0]
    boxes = result.boxes
    if boxes is None or boxes.xyxy is None:
        return []

    frame_height, frame_width = frame.shape[:2]
    detections: List[FaceDetection] = []
    for xyxy, conf in zip(boxes.xyxy.tolist(), boxes.conf.tolist()):
        x1, y1, x2, y2 = xyxy
        x1 = max(0, min(int(x1), frame_width - 1))
        y1 = max(0, min(int(y1), frame_height - 1))
        x2 = max(0, min(int(x2), frame_width))
        y2 = max(0, min(int(y2), frame_height))
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        if w == 0 or h == 0:
            continue
        detections.append(
            FaceDetection(bbox=(x1, y1, w, h), confidence=float(conf))
        )

    return detections


@profile_time("calculate_blur_score")
def calculate_blur_score(frame: np.ndarray) -> float:
    """Calculate Laplacian variance for blur detection."""
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        return 0.0
    try:
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())
    except Exception as exc:  # pragma: no cover - defensive path
        logger.warning("Blur score calculation failed: %s", exc)
        return 0.0


def categorize_pose(face_bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int]) -> str:
    """Categorize pose: portrait, profile, full-body, etc."""
    frame_height, frame_width = frame_shape[:2]
    if frame_height == 0 or frame_width == 0:
        return "unknown"

    x, y, w, h = face_bbox
    face_area_ratio = (w * h) / float(frame_width * frame_height)
    aspect_ratio = (w / h) if h else 0.0

    if face_area_ratio >= _PORTRAIT_THRESHOLD:
        return "portrait"
    if face_area_ratio <= _FULL_BODY_THRESHOLD:
        return "full-body"
    if aspect_ratio < _PROFILE_ASPECT_RATIO_THRESHOLD:
        return "profile"
    return "medium-shot"


def _compute_centering_score(frame_shape: Tuple[int, int], face_bbox: Tuple[int, int, int, int]) -> float:
    frame_height, frame_width = frame_shape[:2]
    frame_center_x = frame_width / 2.0
    frame_center_y = frame_height / 2.0
    x, y, w, h = face_bbox
    face_center_x = x + (w / 2.0)
    face_center_y = y + (h / 2.0)
    distance = ((face_center_x - frame_center_x) ** 2 +
                (face_center_y - frame_center_y) ** 2) ** 0.5
    max_distance = ((frame_width ** 2 + frame_height ** 2) ** 0.5) / 2.0
    if max_distance == 0:
        return 0.0
    return 1.0 - min(distance / max_distance, 1.0)


def _compute_face_quality_for_bbox(
    frame: np.ndarray,
    face_bbox: Tuple[int, int, int, int],
    confidence: float,
    clarity_normalization: float,
    size_normalization: float,
) -> float:
    frame_height, frame_width = frame.shape[:2]
    x, y, w, h = face_bbox
    face_area_ratio = (w * h) / float(frame_width * frame_height)
    size_score = min(face_area_ratio / size_normalization, 1.0)

    face_crop = frame[y:y + h, x:x + w] if w > 0 and h > 0 else frame
    clarity_score = calculate_blur_score(face_crop)
    clarity_score = min(clarity_score / max(clarity_normalization, 1.0), 1.0)

    centering_score = _compute_centering_score(frame.shape, face_bbox)
    confidence_score = min(max(confidence, 0.0), 1.0)

    quality = (size_score * 0.5 + clarity_score * 0.3 + centering_score * 0.2) * 10.0
    return max(min(quality * confidence_score, 10.0), 0.0)


def compute_face_quality_score_from_faces(
    frame: np.ndarray,
    faces: List[FaceDetection],
    clarity_normalization: float = _CLARITY_NORMALIZATION,
    size_normalization: float = _FACE_SIZE_NORMALIZATION,
) -> float:
    """Combined score: face size, clarity, centering."""
    if not faces:
        return 0.0
    primary_face = max(faces, key=lambda face: face.area)
    return _compute_face_quality_for_bbox(
        frame,
        primary_face.bbox,
        primary_face.confidence,
        clarity_normalization,
        size_normalization,
    )


def compute_face_quality_score(frame: np.ndarray) -> float:
    """Combined score: face size, clarity, centering."""
    faces = detect_faces(frame)
    return compute_face_quality_score_from_faces(frame, faces)
