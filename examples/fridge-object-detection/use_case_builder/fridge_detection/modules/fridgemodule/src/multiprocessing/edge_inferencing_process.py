import os
import logging
import signal

from src.frame_processor.config import EdgeInferenceConfig
from src.frame_processor.edge_inferencing import EdgeInferencing
from src.multiprocessing.queue import ProcessQueue


class EdgeInferencingProcess:
    """
    Class for edge inferencing process.
    """

    def __init__(self) -> None:
        """
        Initialize the edge inferencing process.
        """
        self.logger = logging.getLogger(__name__)

    def run(self, edge_inference_config: EdgeInferenceConfig, frame_queue: ProcessQueue, inference_queue: ProcessQueue) -> None:
        """
        Entrypoint for the edge inferencing process.

        @param
            queue (ProcessQueue): A multiprocessing queue used to communicate with edge inferencing process
        """
        self.logger.info("Starting edge inferencing process...")
        self._process(edge_inference_config, frame_queue, inference_queue)
        self.logger.info("Edge inferencing process started...")

    def _process(
            self,
            edge_inference_config: EdgeInferenceConfig,
            frame_queue: ProcessQueue,
            inference_queue: ProcessQueue) -> None:
        """
        Run edge inferencing.

        @param
            queue (ProcessQueue): A multiprocessing queue used to communicate with edge inferencing process
        """
        try:
            edge_inferencing = EdgeInferencing(edge_inference_config, frame_queue, inference_queue)
            edge_inferencing.run()
        except Exception as e:
            self.logger.critical(f"Failed to start edge inferencing process: {e}")
            self.logger.exception(e)
            self.logger.critical(f"Killing main process (pid = {os.getppid()})")
            os.kill(os.getppid(), signal.SIGTERM)
