"""Decode YOLO detection or COCO-pose output into person observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preprocess import ImageTransform


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2
    score: float
    keypoints: np.ndarray | None = None  # COCO shape (17, 3): x, y, confidence


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _rows(output: np.ndarray) -> np.ndarray:
    array = np.asarray(output)
    if array.ndim == 3:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Unexpected YOLO output shape: {np.asarray(output).shape}")
    # Ultralytics COCO exports channels-first (84xN detect / 56xN pose).
    # Explicit channel counts also make one-candidate fixture/video outputs work.
    if array.shape[0] in (56, 84):
        array = array.T
    return array


def _unletterbox_xyxy(box: np.ndarray, transform: ImageTransform) -> tuple[float, float, float, float]:
    box[[0, 2]] = (box[[0, 2]] - transform.pad_x) / transform.scale
    box[[1, 3]] = (box[[1, 3]] - transform.pad_y) / transform.scale
    box[[0, 2]] = np.clip(box[[0, 2]], 0, transform.original_width)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, transform.original_height)
    return tuple(float(value) for value in box)


def postprocess(
    output: np.ndarray,
    transform: ImageTransform,
    confidence_threshold: float = 0.45,
    nms_threshold: float = 0.45,
) -> list[Detection]:
    candidates: list[Detection] = []
    for row in _rows(output):
        if row.size == 56:  # YOLOv8 pose: xywh, person score, 17*(x,y,confidence)
            score = float(row[4])
            keypoint_values = row[5:56]
        elif row.size >= 6:  # YOLOv8 detect: xywh followed by class scores
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            if class_id != 0:
                continue
            score = float(class_scores[class_id])
            keypoint_values = None
        else:
            continue
        if score < confidence_threshold:
            continue

        cx, cy, width, height = (float(value) for value in row[:4])
        box = np.array([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2], dtype=float)
        mapped_box = _unletterbox_xyxy(box, transform)
        if mapped_box[2] <= mapped_box[0] or mapped_box[3] <= mapped_box[1]:
            continue

        keypoints = None
        if keypoint_values is not None:
            keypoints = np.asarray(keypoint_values, dtype=float).reshape(17, 3).copy()
            keypoints[:, 0] = np.clip((keypoints[:, 0] - transform.pad_x) / transform.scale, 0, transform.original_width)
            keypoints[:, 1] = np.clip((keypoints[:, 1] - transform.pad_y) / transform.scale, 0, transform.original_height)
        candidates.append(Detection(mapped_box, score, keypoints))

    selected: list[Detection] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if all(box_iou(candidate.box, kept.box) <= nms_threshold for kept in selected):
            selected.append(candidate)
    return selected
