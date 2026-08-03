import unittest

import numpy as np

from src.detector import Detector


class _Info:
    def __init__(self, name, shape=None):
        self.name = name
        self.shape = shape


class _Session:
    def get_inputs(self):
        return [_Info("images", [1, 3, 320, 320])]

    def get_outputs(self):
        return [_Info("output0")]

    def run(self, names, feed):
        assert names == ["output0"]
        assert feed["images"].shape == (1, 3, 320, 320)
        return [np.ones((1, 84, 2), dtype=np.float32)]


class DetectorTests(unittest.TestCase):
    def test_injected_session_uses_model_shape(self):
        detector = Detector("unused.onnx", session=_Session())
        self.assertEqual(detector.input_size, (320, 320))
        output = detector.infer(np.zeros((1, 3, 320, 320), dtype=np.float32))
        self.assertEqual(output.shape, (1, 84, 2))


if __name__ == "__main__":
    unittest.main()
