import unittest

from src.config import AppConfig, _update_dataclass, _validate


class ConfigTests(unittest.TestCase):
    def test_nested_override(self):
        config = AppConfig()
        _update_dataclass(config, {"model": {"threads": 3}, "camera": {"width": 1280}})
        self.assertEqual(config.model.threads, 3)
        self.assertEqual(config.camera.width, 1280)

    def test_unknown_key_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "Unknown configuration"):
            _update_dataclass(AppConfig(), {"model": {"typo": True}})

    def test_unsafe_action_is_rejected(self):
        config = AppConfig()
        config.gestures.actions["wave"] = "arm_motors"
        with self.assertRaisesRegex(ValueError, "Unsafe/unknown"):
            _validate(config)

    def test_zero_inference_stride_is_rejected(self):
        config = AppConfig()
        config.model.inference_stride = 0
        with self.assertRaisesRegex(ValueError, "inference_stride"):
            _validate(config)


if __name__ == "__main__":
    unittest.main()
