"""Agni command-line vision loop. Output is safe JSON intent, not motor control."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from .camera import Camera
from .config import load_config
from .controller import FollowController
from .detector import Detector
from .gestures import GestureRecognizer
from .postprocess import postprocess
from .preprocess import preprocess
from .tracker import PersonTracker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agni person following vision prototype")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--camera", choices=("picamera2", "opencv"))
    parser.add_argument("--source", help="OpenCV camera index or video path")
    parser.add_argument("--display", action="store_true", help="Show an annotated local preview")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--target-id", type=int, help="Explicit session track ID to follow")
    target.add_argument("--auto-follow", action="store_true", help="Lock the largest first visible person")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 means run forever)")
    return parser


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    backend = args.camera or config.camera.backend
    source = args.source if args.source is not None else config.camera.source
    model_path = Path(args.config).resolve().parent.parent / config.model.path
    if not model_path.exists():
        model_path = Path(config.model.path)
    detector = Detector(model_path, config.model.threads)
    tracker = PersonTracker(**asdict(config.tracking))
    gestures = GestureRecognizer(
        config.gestures.hold_frames,
        config.gestures.cooldown_seconds,
        config.gestures.keypoint_threshold,
    )
    controller = FollowController(**asdict(config.controller), gesture_actions=config.gestures.actions)
    previous = time.monotonic()

    with Camera(backend, source, config.camera.width, config.camera.height) as camera:
        for frame_number in range(1, args.max_frames + 1 if args.max_frames else 2**63):
            try:
                frame = camera.read()
            except EOFError:
                break
            if frame_number % config.model.inference_stride != 0:
                continue
            tensor, transform = preprocess(frame, detector.input_size)
            output = detector.infer(tensor)
            detections = postprocess(output, transform, config.model.confidence, config.model.iou_threshold)
            tracks = tracker.update(detections, (frame.shape[1], frame.shape[0]))
            if not controller.following and controller.target_id is None:
                if args.target_id is not None and any(track.track_id == args.target_id for track in tracks):
                    controller.start_follow(args.target_id)
                elif args.auto_follow and tracks:
                    largest = max(tracks, key=lambda track: (track.box[2] - track.box[0]) * (track.box[3] - track.box[1]))
                    controller.start_follow(largest.track_id)
            events = gestures.update(tracks)
            intent = controller.update(tracks, events, (frame.shape[1], frame.shape[0]))
            now = time.monotonic()
            fps = 1.0 / max(now - previous, 1e-9)
            previous = now
            print(json.dumps({
                "frame": frame_number,
                "fps": round(fps, 1),
                "track_ids": [track.track_id for track in tracks],
                "gestures": [{"track_id": event.track_id, "name": event.name} for event in events],
                "intent": json.loads(intent.to_json()),
            }), flush=True)

            if args.display:
                try:
                    import cv2
                except ImportError as exc:
                    raise RuntimeError("--display requires OpenCV") from exc
                for track in tracks:
                    x1, y1, x2, y2 = map(int, track.box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 120), 2)
                    cv2.putText(frame, f"person {track.track_id}", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 120), 2)
                cv2.imshow("Agni dry-run", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    if args.display:
        import cv2
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
