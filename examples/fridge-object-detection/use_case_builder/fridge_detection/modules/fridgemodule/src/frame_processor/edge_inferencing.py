import base64
from collections import Counter
import cv2
import json
import logging
import numpy as np
import requests
from typing import List

from src.frame_processor.config import EdgeInferenceConfig
from src.common.utils import retry_session
from src.frame_processor.model import InferenceRequest, InferenceResultRaw, InferenceResultTransformed
from src.multiprocessing.queue import ProcessQueue

# Disable the warning from urllib3.connectionpool
logging.getLogger("urllib3").setLevel(logging.ERROR)


class EdgeInferencing:
    """
    Class for orchestrating inference on edge.
    """

    def __init__(
            self,
            edge_inference_config: EdgeInferenceConfig,
            input_queue: ProcessQueue,
            output_queue: ProcessQueue) -> None:
        """
        Initialize EdgeInferencing.
        """
        self.logger = logging.getLogger(__name__)

        self.frame_queue = input_queue
        self.inference_queue = output_queue

        self.edge_inference_config = edge_inference_config

    def run(self):
        """
        Run inference on edge.
        """
        while True:
            frame = self.frame_queue.get_item()
            if frame is not None:
                pre_processed_frame = self._pre_process(frame)
                request = self._prepare_request(pre_processed_frame)
                response = self._invoke_model_endpoint(request)
                if response is None:
                    self.logger.error("Unable to process frame, skipping...")
                    continue
                inference_results = self._post_process(response)
                for inference_result in inference_results:
                    self.inference_queue.add_item(inference_result)

    def _pre_process(self, frame: np.ndarray) -> np.ndarray:
        """
        Pre-process frame before inference.
        """
        # TODO: Add generic (not model specific) pre-processing logic here if applicable
        return frame

    def _prepare_request(self, frame: np.ndarray) -> InferenceRequest:
        """
        Prepare request for inference.
        """
        jpg = cv2.imencode(".jpg", frame)[1].tobytes()
        frame_b64 = base64.b64encode(jpg).decode("utf-8")

        request = InferenceRequest(frame_b64)
        return request

    def _invoke_model_endpoint(self, request: InferenceRequest) -> List[List[InferenceResultRaw]]:
        """
        Invoke model endpoint.
        """
        payload = json.dumps(
            {
                "images": [
                    request.frame
                ]
            }
        )

        headers = {
            'Content-Type': 'application/json'
        }

        try:
            session = retry_session()
            response = session.post(self.edge_inference_config.model_endpoint, headers=headers, data=payload)
            if response.status_code == 200:
                batch_prediction = json.loads(response.text)
            else:
                self.logger.error(f"Invalid response from inference module {response.status_code} {response.text}")
                return None
        except requests.exceptions.RequestException:
            self.logger.exception("Error connecting to inference module")
            return None
        except json.JSONDecodeError:
            self.logger.exception("Error parsing response from inference module")
            return None
        except Exception:
            self.logger.exception("Error invoking model endpoint")
            return None

        inference_batch_result = []
        for prediction in batch_prediction:
            result = [
                InferenceResultRaw(item["box"], item["label"], item["score"])
                for item in prediction
            ]
            inference_batch_result.append(result)
        return inference_batch_result

    def _post_process(self, inference_batch_result: List[List[InferenceResultRaw]]) -> List[InferenceResultTransformed]:
        """
        Post-process inference results.
        """
        # Add generic (not model specific) post-processing logic here if applicable

        transformed_result = [
            InferenceResultTransformed(inference_result, dict(Counter(item.label for item in inference_result)))
            for inference_result in inference_batch_result
        ]
        return transformed_result
