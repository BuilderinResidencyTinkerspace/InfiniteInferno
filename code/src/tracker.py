"""Lightweight IoU/centroid person tracker with session-local IDs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import hypot

import numpy as np

from .postprocess import Detection, box_iou


@dataclass(frozen=True)
class Track:
    track_id: int
    box: tuple[float, float, float, float]
    score: float
    keypoints: np.ndarray | None
    hits: int = 1
    missed: int = 0
    confirmed: bool = False

    @property
    def center(self) -> tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2)


class PersonTracker:
    def __init__(self, max_missed: int = 12, min_hits: int = 2, iou_threshold: float = 0.2, max_center_distance: float = 0.18):
        self.max_missed = max_missed
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_center_distance = max_center_distance
        self._tracks: dict[int, Track] = {}
        self._next_id = 1

    def update(self, detections: list[Detection], frame_size: tuple[int, int]) -> list[Track]:
        diagonal = hypot(*frame_size)
        candidates: list[tuple[float, int, int]] = []
        for track_id, track in self._tracks.items():
            for detection_index, detection in enumerate(detections):
                overlap = box_iou(track.box, detection.box)
                dcx = (detection.box[0] + detection.box[2]) / 2 - track.center[0]
                dcy = (detection.box[1] + detection.box[3]) / 2 - track.center[1]
                distance = hypot(dcx, dcy) / max(diagonal, 1.0)
                if overlap >= self.iou_threshold or distance <= self.max_center_distance:
                    candidates.append((overlap - distance * 0.25, track_id, detection_index))

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for _, track_id, detection_index in sorted(candidates, reverse=True):
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            old = self._tracks[track_id]
            detection = detections[detection_index]
            hits = old.hits + 1
            self._tracks[track_id] = Track(track_id, detection.box, detection.score, detection.keypoints, hits, 0, hits >= self.min_hits)
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)

        for track_id, track in list(self._tracks.items()):
            if track_id not in matched_tracks:
                missed = track.missed + 1
                if missed > self.max_missed:
                    del self._tracks[track_id]
                else:
                    self._tracks[track_id] = replace(track, missed=missed)

        for index, detection in enumerate(detections):
            if index in matched_detections:
                continue
            track_id = self._next_id
            self._next_id += 1
            self._tracks[track_id] = Track(
                track_id, detection.box, detection.score, detection.keypoints,
                confirmed=self.min_hits <= 1,
            )

        return sorted((track for track in self._tracks.values() if track.confirmed and track.missed == 0), key=lambda item: item.track_id)
