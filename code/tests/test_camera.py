import unittest
from unittest.mock import MagicMock, patch

from src.camera import Camera


class CameraTests(unittest.TestCase):
    def test_unknown_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown camera backend"):
            Camera(backend="invalid")

    def test_opencv_read_failure_is_an_eof(self):
        fake_cv2 = MagicMock()
        capture = fake_cv2.VideoCapture.return_value
        capture.isOpened.return_value = True
        capture.read.return_value = (False, None)
        fake_cv2.COLOR_BGR2RGB = 1
        with patch.dict("sys.modules", {"cv2": fake_cv2}):
            camera = Camera(backend="opencv", source=0)
            with self.assertRaises(EOFError):
                camera.read()
            camera.release()
        capture.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
