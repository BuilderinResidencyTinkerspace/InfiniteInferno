"""YOLO letterbox preprocessing without requiring OpenCV."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ImageTransform:
    scale: float
    pad_x: int
    pad_y: int
    input_width: int
    input_height: int
    original_width: int
    original_height: int


def _resize_nearest(image: np.ndarray, width: int, height: int) -> np.ndarray:
    y = np.minimum((np.arange(height) * image.shape[0] / height).astype(int), image.shape[0] - 1)
    x = np.minimum((np.arange(width) * image.shape[1] / width).astype(int), image.shape[1] - 1)
    return image[y[:, None], x[None, :]]


def preprocess(frame: np.ndarray, input_size: tuple[int, int] = (640, 640)) -> tuple[np.ndarray, ImageTransform]:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected an HxWx3 image")
    input_width, input_height = input_size
    height, width = frame.shape[:2]
    scale = min(input_width / width, input_height / height)
    resized_width, resized_height = max(1, round(width * scale)), max(1, round(height * scale))

    try:
        import cv2
        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        resized = _resize_nearest(frame, resized_width, resized_height)

    canvas = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
    pad_x = (input_width - resized_width) // 2
    pad_y = (input_height - resized_height) // 2
    canvas[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = resized
    tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return tensor, ImageTransform(scale, pad_x, pad_y, input_width, input_height, width, height)
