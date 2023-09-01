import os
import logging
import signal
from src.frame_processor.inference_result_handler import InferenceResultHandler


class InferenceResultHandlerProcess:
    """
    Class for process that handles inference result.
    """

    def __init__(self) -> None:
        """
        Initialize the inference result handler process.
        """
        self.logger = logging.getLogger(__name__)

    def run(self, inference_result_handler: InferenceResultHandler) -> None:
        """
        Entrypoint for the inference result handler process.

        @param
            queue (ProcessQueue): A multiprocessing queue used to communicate with edge inferencing process
        """
        self._process(inference_result_handler)

    def _process(self, inference_result_handler: InferenceResultHandler) -> None:
        """
        Run edge inferencing.

        @param
            queue (ProcessQueue): A multiprocessing queue used to communicate with edge inferencing process
        """
        try:
            self.logger.info("Starting inference result handler process...")
            inference_result_handler.run()
        except Exception as e:
            self.logger.critical(f"Failed to start inference result handler process: {e}")
            self.logger.exception(e)
            self.logger.critical(f"Killing main process (pid = {os.getppid()})")
            os.kill(os.getppid(), signal.SIGTERM)
