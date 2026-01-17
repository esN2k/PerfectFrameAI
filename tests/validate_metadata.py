#!/usr/bin/env python3
"""
Validate metadata JSON files produced by PerfectFrameAI.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2


REQUIRED_KEYS = {
    "frame_id",
    "timestamp",
    "aesthetic_score",
    "face_quality_score",
    "blur_score",
    "composite_score",
    "faces_detected",
    "face_bounding_boxes",
    "pose_category",
    "is_blurry",
}

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def _find_image_for_metadata(metadata_path: Path) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = metadata_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def _iter_metadata_paths(paths: Iterable[Path]) -> Iterable[Path]:
    for root in paths:
        if root.is_file() and root.suffix == ".json":
            yield root
        if root.is_dir():
            yield from root.rglob("*.json")


def _validate_entry(entry: dict, metadata_path: Path,
                    blur_threshold: float,
                    weights: tuple[float, float, float]) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_KEYS.difference(entry.keys())
    if missing:
        errors.append(f"Missing keys: {sorted(missing)}")
        return errors

    aesthetic = float(entry["aesthetic_score"])
    face_quality = float(entry["face_quality_score"])
    blur_score = float(entry["blur_score"])
    composite = float(entry["composite_score"])
    faces_detected = int(entry["faces_detected"])
    bboxes = entry["face_bounding_boxes"]

    if not (1.0 <= aesthetic <= 10.0):
        errors.append(f"aesthetic_score out of range: {aesthetic}")
    if not (0.0 <= face_quality <= 10.0):
        errors.append(f"face_quality_score out of range: {face_quality}")
    if blur_score <= 0.0:
        errors.append(f"blur_score out of range: {blur_score}")
    if faces_detected < 0:
        errors.append(f"faces_detected out of range: {faces_detected}")
    if not isinstance(bboxes, list):
        errors.append("face_bounding_boxes is not a list")

    weight_total = sum(weights)
    if weight_total > 0:
        sharpness_score = min(blur_score / max(blur_threshold, 1.0), 1.0) * 10.0
        expected = (
            aesthetic * weights[0] +
            face_quality * weights[1] +
            sharpness_score * weights[2]
        ) / weight_total
        if abs(expected - composite) > 0.5:
            errors.append(f"composite_score mismatch: {composite} vs {expected:.3f}")

    image_path = _find_image_for_metadata(metadata_path)
    if image_path:
        image = cv2.imread(str(image_path))
        if image is None:
            errors.append(f"Cannot read image for {metadata_path.name}")
        else:
            height, width = image.shape[:2]
            for bbox in bboxes:
                if not isinstance(bbox, list) or len(bbox) != 4:
                    errors.append(f"Invalid bbox format: {bbox}")
                    continue
                x, y, w, h = [int(v) for v in bbox]
                if x < 0 or y < 0 or w < 0 or h < 0:
                    errors.append(f"Negative bbox values: {bbox}")
                if x + w > width or y + h > height:
                    errors.append(f"BBox out of bounds: {bbox} for {width}x{height}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate metadata JSON files.")
    parser.add_argument("paths", nargs="+", type=Path,
                        help="Paths to metadata files or directories.")
    parser.add_argument("--blur-threshold", type=float, default=100.0,
                        help="Blur threshold used for composite score.")
    parser.add_argument("--weights", type=float, nargs=3, default=(0.5, 0.35, 0.15),
                        metavar=("AESTHETIC", "FACE", "SHARPNESS"),
                        help="Composite score weights.")
    args = parser.parse_args()

    failures = 0
    for metadata_path in _iter_metadata_paths(args.paths):
        try:
            entry = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"{metadata_path}: invalid JSON ({exc})")
            failures += 1
            continue
        errors = _validate_entry(entry, metadata_path,
                                 args.blur_threshold, tuple(args.weights))
        if errors:
            failures += 1
            print(f"{metadata_path}:")
            for err in errors:
                print(f"  - {err}")

    if failures:
        print(f"Validation failed for {failures} file(s).")
        return 1
    print("All metadata files valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
