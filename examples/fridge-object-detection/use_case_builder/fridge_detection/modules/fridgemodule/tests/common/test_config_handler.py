import os
import unittest
from unittest import mock
from unittest.mock import patch
import uuid

from src.common.config_handler import ConfigHandler
from src.frame_capture.config import FrameCaptureConfig
from src.frame_processor.config import EdgeInferenceConfig


class TestConfigHandler(unittest.TestCase):
    @patch("src.common.config_handler.ConfigHandler._read_from_env_variable",
           side_effect=Exception("test"),)
    def test_get_config_raises_error_if_config_source_is_not_supported(self, _):
        config_handler = ConfigHandler(config_source="test")
        self.assertRaises(
            ValueError,
            config_handler._get_config,
            str(uuid.uuid1()),
        )

    def test_get_config_raises_error_if_config_not_found(self):
        config_handler = ConfigHandler()
        self.assertRaises(
            Exception,
            config_handler._get_config,
            str(uuid.uuid1()),
        )

    def test_get_config_returns_config_if_found(self):
        config_handler = ConfigHandler()
        with patch(
            "src.common.config_handler.ConfigHandler._read_from_env_variable",
            return_value="test",
        ):
            self.assertEqual(config_handler._get_config("test"), "test")

    def test_read_from_env_variable_raises_error_if_config_not_found(self):
        config_handler = ConfigHandler()
        self.assertRaises(
            ValueError, config_handler._read_from_env_variable, str(uuid.uuid1())
        )

    def test_read_from_env_variable_returns_config_if_found(self):
        config_handler = ConfigHandler()
        with patch(
            "src.common.config_handler.os.environ.get",
            return_value="test",
        ):
            self.assertEqual(config_handler._read_from_env_variable("test"), "test")

    @mock.patch.dict(os.environ, {"CAMERA_PATH": "test_camera_path"})
    @mock.patch.dict(os.environ, {"FRAME_RATE": "1"})
    def test_get_frame_capture_config(self):
        config_handler = ConfigHandler()
        frame_capture_config = config_handler.get_frame_capture_config()
        self.assertIsInstance(frame_capture_config, FrameCaptureConfig)

    @mock.patch.dict(os.environ, {"MODEL_ENDPOINT": "http://localhost:8081/score"})
    def test_get_edge_inference_config(self):
        config_handler = ConfigHandler()
        edge_inference_config = config_handler.get_edge_inference_config()
        self.assertIsInstance(edge_inference_config, EdgeInferenceConfig)
