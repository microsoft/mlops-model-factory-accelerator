import logging
import time

from src.common import constants
from src.multiprocessing.queue import ProcessQueue

from abc import ABC, abstractmethod


class InferenceResultHandler(ABC):
    def __init__(self, queue: ProcessQueue):
        self.logger = logging.getLogger(__name__)
        self.queue = queue

    @abstractmethod
    def run(self):
        pass


class DefaultInferenceResultHandler(InferenceResultHandler):
    def run(self):
        while True:
            inference_result = self.queue.get_item()
            if inference_result is not None:
                self.logger.info(f"Inference Result: {inference_result.object_cnt}")


class E2ETestInferenceResultHandler(InferenceResultHandler):
    def run(self):
        while True:
            time.sleep(100000)

    def get_inference_result(self):
        start_time = time.time()
        max_time = constants.E2E_TEST_TIMEOUT

        while True:
            inference_result = self.queue.get_item()
            if inference_result is not None:
                self.logger.info(f"Inference Result E2E Test: {inference_result.object_cnt}")
                return inference_result.object_cnt

            elapsed_time = time.time() - start_time
            if elapsed_time > max_time:
                break

        self.logger.info("Timeout occurred while waiting for inference result")
        return None
