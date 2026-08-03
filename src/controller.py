"""Convert target geometry and gestures into bounded, autopilot-agnostic intents."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

from .gestures import GestureEvent
from .tracker import Track


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


@dataclass(frozen=True)
class FlightIntent:
    yaw: float = 0.0
    forward: float = 0.0
    vertical: float = 0.0
    action: str = "hover"
    target_id: int | None = None
    reason: str = "safe default"

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


class FollowController:
    def __init__(
        self,
        target_area_ratio: float = 0.18,
        deadband_x: float = 0.08,
        deadband_y: float = 0.10,
        max_yaw: float = 0.35,
        max_forward: float = 0.30,
        max_vertical: float = 0.25,
        target_timeout_seconds: float = 0.75,
        gesture_actions: dict[str, str] | None = None,
    ):
        self.target_area_ratio = target_area_ratio
        self.deadband_x = deadband_x
        self.deadband_y = deadband_y
        self.max_yaw = max_yaw
        self.max_forward = max_forward
        self.max_vertical = max_vertical
        self.target_timeout_seconds = target_timeout_seconds
        self.gesture_actions = gesture_actions or {}
        self.target_id: int | None = None
        self.following = False
        self._last_seen: float | None = None

    def _apply_gestures(self, gestures: list[GestureEvent]) -> str | None:
        for event in gestures:
            action = self.gesture_actions.get(event.name)
            if action == "start_follow":
                # Once locked, another bystander cannot silently steal control.
                if self.target_id is None or self.target_id == event.track_id:
                    self.target_id, self.following = event.track_id, True
                    self._last_seen = None
            elif action in {"stop_follow", "hover"}:
                if self.target_id is None or self.target_id == event.track_id:
                    self.following = False
            elif action == "land" and (self.target_id is None or self.target_id == event.track_id):
                self.following = False
                return "land"
        return None

    def start_follow(self, track_id: int) -> None:
        """Explicit operator selection for detection-only models."""
        self.target_id = track_id
        self.following = True
        self._last_seen = None

    def update(self, tracks: list[Track], gestures: list[GestureEvent], frame_size: tuple[int, int], now: float | None = None) -> FlightIntent:
        now = time.monotonic() if now is None else now
        immediate = self._apply_gestures(gestures)
        if immediate == "land":
            return FlightIntent(action="land", target_id=self.target_id, reason="confirmed gesture")
        if not self.following or self.target_id is None:
            return FlightIntent(target_id=self.target_id, reason="follow mode disabled")

        target = next((track for track in tracks if track.track_id == self.target_id), None)
        if target is None:
            elapsed = float("inf") if self._last_seen is None else now - self._last_seen
            reason = "target lost" if elapsed >= self.target_timeout_seconds else "target temporarily occluded"
            return FlightIntent(target_id=self.target_id, reason=reason)
        self._last_seen = now

        frame_width, frame_height = frame_size
        cx, cy = target.center
        error_x = (cx - frame_width / 2) / (frame_width / 2)
        error_y = (frame_height / 2 - cy) / (frame_height / 2)
        area = (target.box[2] - target.box[0]) * (target.box[3] - target.box[1])
        area_ratio = area / max(1, frame_width * frame_height)
        distance_error = (self.target_area_ratio - area_ratio) / self.target_area_ratio

        yaw = 0.0 if abs(error_x) <= self.deadband_x else _clamp(error_x * self.max_yaw, self.max_yaw)
        vertical = 0.0 if abs(error_y) <= self.deadband_y else _clamp(error_y * self.max_vertical, self.max_vertical)
        forward = 0.0 if abs(distance_error) <= 0.1 else _clamp(distance_error * self.max_forward, self.max_forward)
        return FlightIntent(yaw, forward, vertical, "follow", target.track_id, "target locked")
