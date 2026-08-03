"""Debounced gestures from standard 17-point COCO pose keypoints."""

from __future__ import annotations

import time
from dataclasses import dataclass

from .tracker import Track


@dataclass(frozen=True)
class GestureEvent:
    track_id: int
    name: str
    timestamp: float


class GestureRecognizer:
    # COCO: nose=0, shoulders=5/6, wrists=9/10.
    def __init__(self, hold_frames: int = 4, cooldown_seconds: float = 2.0, keypoint_threshold: float = 0.35):
        self.hold_frames = hold_frames
        self.cooldown_seconds = cooldown_seconds
        self.keypoint_threshold = keypoint_threshold
        self._state: dict[int, tuple[str | None, int]] = {}
        self._last_emitted: dict[tuple[int, str], float] = {}

    def classify(self, track: Track) -> str | None:
        points = track.keypoints
        if points is None or points.shape != (17, 3):
            return None
        nose, left_shoulder, right_shoulder, left_wrist, right_wrist = points[[0, 5, 6, 9, 10]]
        if min(nose[2], left_shoulder[2], right_shoulder[2]) < self.keypoint_threshold:
            return None
        left_up = left_wrist[2] >= self.keypoint_threshold and left_wrist[1] < left_shoulder[1]
        right_up = right_wrist[2] >= self.keypoint_threshold and right_wrist[1] < right_shoulder[1]
        if left_up and right_up:
            return "both_hands_up"
        if right_up:
            return "right_hand_up"
        if left_up:
            return "left_hand_up"
        return None

    def update(self, tracks: list[Track], now: float | None = None) -> list[GestureEvent]:
        now = time.monotonic() if now is None else now
        events: list[GestureEvent] = []
        visible_ids = {track.track_id for track in tracks}
        for stale_id in set(self._state) - visible_ids:
            del self._state[stale_id]
        for track in tracks:
            gesture = self.classify(track)
            previous, count = self._state.get(track.track_id, (None, 0))
            count = count + 1 if gesture is not None and gesture == previous else (1 if gesture else 0)
            self._state[track.track_id] = (gesture, count)
            if gesture and count == self.hold_frames:
                key = (track.track_id, gesture)
                if now - self._last_emitted.get(key, float("-inf")) >= self.cooldown_seconds:
                    self._last_emitted[key] = now
                    events.append(GestureEvent(track.track_id, gesture, now))
        return events
