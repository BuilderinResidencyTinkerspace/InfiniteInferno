"""Lazy camera backends so tests do not require Raspberry Pi packages."""

from __future__ import annotations

from typing import Any, Self

import numpy as np


class Camera:
    def __init__(self, backend: str = "picamera2", source: str | int = 0, width: int = 640, height: int = 480):
        self.backend = backend
        self.device: Any
        if backend == "picamera2":
            try:
                from picamera2 import Picamera2
            except ImportError as exc:
                raise RuntimeError("Picamera2 is unavailable; install python3-picamera2 or use --camera opencv") from exc
            self.device = Picamera2()
            config = self.device.create_video_configuration(main={"size": (width, height), "format": "RGB888"})
            self.device.configure(config)
            self.device.start()
        elif backend == "opencv":
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError("OpenCV is unavailable; install python3-opencv") from exc
            parsed_source: str | int = int(source) if str(source).isdigit() else str(source)
            self.device = cv2.VideoCapture(parsed_source)
            self.device.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.device.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if not self.device.isOpened():
                raise RuntimeError(f"Could not open camera/video source {source!r}")
        else:
            raise ValueError(f"Unknown camera backend: {backend}")

    def read(self) -> np.ndarray:
        if self.backend == "picamera2":
            return np.asarray(self.device.capture_array())
        ok, frame = self.device.read()
        if not ok or frame is None:
            raise EOFError("Camera/video stream ended")
        # OpenCV captures BGR; the exported YOLO model expects RGB.
        import cv2
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def release(self) -> None:
        if self.backend == "picamera2":
            self.device.stop()
        else:
            self.device.release()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.release()
