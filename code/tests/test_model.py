import unittest

import numpy as np

from src.postprocess import postprocess
from src.preprocess import preprocess


class ModelPipelineTests(unittest.TestCase):
    def test_letterbox_preserves_geometry_and_maps_detection_back(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tensor, transform = preprocess(frame)
        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertEqual(transform.pad_y, 80)

        output = np.zeros((1, 84, 1), dtype=np.float32)
        output[0, :4, 0] = [320, 320, 320, 240]
        output[0, 4, 0] = 0.9
        detections = postprocess(output, transform)
        self.assertEqual(len(detections), 1)
        np.testing.assert_allclose(detections[0].box, (160, 120, 480, 360), atol=1)

    def test_nms_removes_overlapping_lower_score_person(self):
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        _, transform = preprocess(frame)
        output = np.zeros((1, 84, 2), dtype=np.float32)
        output[0, :4, 0] = [320, 320, 200, 300]
        output[0, :4, 1] = [325, 320, 200, 300]
        output[0, 4, :] = [0.9, 0.7]
        detections = postprocess(output, transform)
        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0].score, 0.9, places=5)

    def test_pose_output_returns_keypoints(self):
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        _, transform = preprocess(frame)
        output = np.zeros((1, 56, 1), dtype=np.float32)
        output[0, :5, 0] = [320, 320, 200, 400, 0.95]
        points = np.tile([320, 300, 0.9], 17)
        output[0, 5:, 0] = points
        detections = postprocess(output, transform)
        self.assertEqual(detections[0].keypoints.shape, (17, 3))


if __name__ == "__main__":
    unittest.main()
