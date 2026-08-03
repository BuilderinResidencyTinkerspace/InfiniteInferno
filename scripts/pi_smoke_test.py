#!/usr/bin/env python3
"""Hardware-free dependency/model smoke test intended to run on the Pi itself."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

from src.detector import Detector


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/yolov8n.onnx")
    args = parser.parse_args()
    detector = Detector(Path(args.model))
    width, height = detector.input_size
    result = detector.infer(np.zeros((1, 3, height, width), dtype=np.float32))
    report = {
        "machine": platform.machine(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "model_input": [width, height],
        "model_output": list(result.shape),
        "status": "ok",
    }
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
