import unittest
from src.frame_processor.config import EdgeInferenceConfig


class TestEdgeInferenceConfig(unittest.TestCase):
    def test_edge_inference_config_init_with_default_values(self):
        config = EdgeInferenceConfig("http://localhost:8081/score")
        self.assertEqual(config.model_endpoint, "http://localhost:8081/score")
