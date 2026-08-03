"""Typed configuration with conservative defaults."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    path: str = "models/yolov8n.onnx"
    confidence: float = 0.45
    iou_threshold: float = 0.45
    threads: int = 4
    inference_stride: int = 2


@dataclass
class CameraConfig:
    backend: str = "picamera2"
    source: str | int = 0
    width: int = 640
    height: int = 480


@dataclass
class TrackingConfig:
    max_missed: int = 12
    min_hits: int = 2
    iou_threshold: float = 0.20
    max_center_distance: float = 0.18


@dataclass
class GestureConfig:
    hold_frames: int = 4
    cooldown_seconds: float = 2.0
    keypoint_threshold: float = 0.35
    actions: dict[str, str] = field(default_factory=lambda: {
        "right_hand_up": "start_follow",
        "left_hand_up": "stop_follow",
        "both_hands_up": "land",
    })


@dataclass
class ControllerConfig:
    target_area_ratio: float = 0.18
    deadband_x: float = 0.08
    deadband_y: float = 0.10
    max_yaw: float = 0.35
    max_forward: float = 0.30
    max_vertical: float = 0.25
    target_timeout_seconds: float = 0.75


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    gestures: GestureConfig = field(default_factory=GestureConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)


def _update_dataclass(instance: Any, values: dict[str, Any]) -> None:
    valid = {item.name: item for item in fields(instance)}
    unknown = set(values) - set(valid)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    for name, value in values.items():
        current = getattr(instance, name)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"Configuration section {name!r} must be a mapping")
            _update_dataclass(current, value)
        else:
            setattr(instance, name, value)


def load_config(path: str | Path) -> AppConfig:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency error is actionable
        raise RuntimeError("PyYAML is required: python -m pip install -r requirements.txt") from exc

    config = AppConfig()
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError("Configuration root must be a mapping")
    _update_dataclass(config, data)
    _validate(config)
    return config


def _validate(config: AppConfig) -> None:
    if not 0.0 < config.model.confidence <= 1.0:
        raise ValueError("model.confidence must be in (0, 1]")
    if config.model.inference_stride < 1:
        raise ValueError("model.inference_stride must be at least 1")
    if config.camera.width <= 0 or config.camera.height <= 0:
        raise ValueError("camera dimensions must be positive")
    if config.tracking.max_missed < 0 or config.tracking.min_hits < 1:
        raise ValueError("tracking max_missed/min_hits are invalid")
    allowed = {"start_follow", "stop_follow", "hover", "land"}
    invalid = set(config.gestures.actions.values()) - allowed
    if invalid:
        raise ValueError(f"Unsafe/unknown gesture actions: {sorted(invalid)}")
