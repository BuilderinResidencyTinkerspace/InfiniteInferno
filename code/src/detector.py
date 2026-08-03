"""Small ONNX Runtime wrapper tuned for Raspberry Pi CPU inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class Detector:
    def __init__(self, model_path: str | Path, threads: int = 4, session: Any | None = None):
        if session is None:
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError("onnxruntime is required: python -m pip install -r requirements.txt") from exc
            path = Path(model_path)
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"ONNX model not found or empty: {path}")
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.intra_op_num_threads = max(1, threads)
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])

        self.session = session
        input_info = session.get_inputs()[0]
        self.input_name = input_info.name
        self.output_name = session.get_outputs()[0].name
        shape = input_info.shape
        self.input_size = (
            int(shape[3]) if isinstance(shape[3], int) else 640,
            int(shape[2]) if isinstance(shape[2], int) else 640,
        )

    def infer(self, image: np.ndarray) -> np.ndarray:
        return np.asarray(self.session.run([self.output_name], {self.input_name: image})[0])
