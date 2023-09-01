import base64
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import logging
import requests

from src.frame_processor.config import EdgeInferenceConfig
from src.frame_processor.edge_inferencing import (
    EdgeInferencing,
    InferenceRequest,
    InferenceResultRaw,
    InferenceResultTransformed
)
from src.multiprocessing.queue import ProcessQueue


class EdgeInferencingTestCase(unittest.TestCase):

    def setUp(self, *args):
        self.edge_inference_config = EdgeInferenceConfig("http://inferencemodule:8081/score")
        self.input_queue = ProcessQueue()
        self.output_queue = ProcessQueue()
        self.edge_inferencing = EdgeInferencing(self.edge_inference_config, self.input_queue, self.output_queue)

    def test_pre_process(self):
        # Arrange
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Act
        pre_processed_frame = self.edge_inferencing._pre_process(frame)

        # Assert
        self.assertIsInstance(pre_processed_frame, np.ndarray)

    def test_prepare_request(self):
        # Arrange
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Act
        request = self.edge_inferencing._prepare_request(frame)

        # Assert
        self.assertIsInstance(request, InferenceRequest)
        self.assertIsInstance(request.frame, str)
        is_base64 = self._is_base64_encoded(request.frame)
        self.assertTrue(is_base64)

    def test_invoke_model_endpoint(self):
        # Arrange
        request = InferenceRequest('frame_data')
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = (
            '[ [ { "box": { "topX": 0.6045256423950195, "topY": 0.3294818115234375, '
            '"bottomX": 0.7843792724609375, "bottomY": 0.7044925944010416 }, '
            '"label": "milk_bottle", "score": 0.9989226460456848 }, '
            '{ "box": { "topX": 0.1614885711669922, "topY": 0.23449081420898438, '
            '"bottomX": 0.41723342895507814, "bottomY": 0.7191886901855469 }, '
            '"label": "carton", "score": 0.9981447458267212 } ], '
            '[ { "box": { "topX": 0.6045256423950195, "topY": 0.3294818115234375, '
            '"bottomX": 0.7843792724609375, "bottomY": 0.7044925944010416 }, '
            '"label": "milk_bottle", "score": 0.9989226460456848 } ] ]'
        )

        # Act
        with patch('requests.Session.post', return_value=mock_response):
            result = self.edge_inferencing._invoke_model_endpoint(request)

        # Assert
        self.assertIsInstance(result[0][0], InferenceResultRaw)

    def test_invoke_model_endpoint_invalid_response(self):
        # Arrange
        request = InferenceRequest('frame_data')
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Invalid response"

        # Act
        with patch('requests.Session.post', return_value=mock_response):
            with self.assertLogs(level=logging.ERROR) as cm:
                result = self.edge_inferencing._invoke_model_endpoint(request)

        # Assert
        self.assertIsNone(result)
        self.assertTrue(any("Error parsing response" in msg for msg in cm.output))

    def test_invoke_model_endpoint_connection_error(self):
        # Arrange
        request = InferenceRequest('frame_data')

        # Act
        with patch('requests.Session.post', side_effect=requests.exceptions.ConnectionError):
            with self.assertLogs(level=logging.ERROR) as cm:
                result = self.edge_inferencing._invoke_model_endpoint(request)

        # Assert
        self.assertIsNone(result)
        self.assertTrue(any("Error connecting to inference module" in msg for msg in cm.output))

    def test_invoke_model_endpoint_response_code_not_200(self):
        # Arrange
        request = InferenceRequest('frame_data')
        mock_response = MagicMock()
        mock_response.status_code = 301
        mock_response.text = "Moved Permanently"

        # Act
        with patch('requests.Session.post', return_value=mock_response):
            with self.assertLogs(level=logging.ERROR) as cm:
                result = self.edge_inferencing._invoke_model_endpoint(request)

        # Assert
        self.assertIsNone(result)
        self.assertTrue(any("Invalid response from inference module 301 Moved Permanently" in msg for msg in cm.output))

    def test_post_process(self):
        # Arrange
        inference_batch_result = [
            [InferenceResultRaw(
                {
                    "topX": 0.1614885711669922,
                    "topY": 0.2344908142089843,
                    "bottomX": 0.4172334289550781,
                    "bottomY": 0.7191886901855469
                },
                "milk_bottle", 0.99)],
            [InferenceResultRaw(
                {
                    "topX": 0.6045256423950195,
                    "topY": 0.3294818115234375,
                    "bottomX": 0.7843792724609375,
                    "bottomY": 0.7044925944010416
                },
                "carton", 0.9)]
        ]

        # Act
        result = self.edge_inferencing._post_process(inference_batch_result)

        # Assert
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], InferenceResultTransformed)
        self.assertEqual(result[0].object_cnt, {"milk_bottle": 1})

    def _is_base64_encoded(self, string):
        try:
            # Attempt to decode the string
            base64.b64decode(string)
            # If decoding succeeds, it is a valid base64 encoded string
            return True
        except base64.Error:
            # If decoding fails, it is not a valid base64 encoded string
            return False


if __name__ == '__main__':
    unittest.main()
