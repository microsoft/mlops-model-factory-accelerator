import unittest
from src.frame_capture.config import FrameCaptureConfig


class TestFrameCaptureConfig(unittest.TestCase):
    def test_frame_capture_config_init_with_default_values(self):
        config = FrameCaptureConfig("test_camera_path", 1)
        self.assertEqual(config.video_source, "test_camera_path")
        self.assertEqual(config.frame_rate, 1)
