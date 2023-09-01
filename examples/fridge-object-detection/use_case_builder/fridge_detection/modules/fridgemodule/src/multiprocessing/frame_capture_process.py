import os
import logging
import signal

from src.frame_capture.config import FrameCaptureConfig
from src.frame_capture.frame_capture import FrameCapture
from src.multiprocessing.queue import ProcessQueue


class FrameCaptureProcess:
    """
    Class for frame capture process.
    """

    def __init__(self) -> None:
        """
        Initialize the frame capture process.
        """
        self.logger = logging.getLogger(__name__)

    def run(self, frame_capture_config: FrameCaptureConfig, frame_queue: ProcessQueue) -> None:
        """
        Entrypoint for the frame capture process.

        @param
            queue (ProcessQueue): A multiprocessing queue used to communicate with frame capture process
        """
        self.logger.info("Starting frame capture process...")
        self._process(frame_capture_config, frame_queue)
        self.logger.info("Frame capture process started...")

    def _process(self, frame_capture_config: FrameCaptureConfig, frame_queue: ProcessQueue) -> None:
        """
        Run frame capture.

        @param
            queue (ProcessQueue): A multiprocessing queue used to communicate with frame capture process
        """
        try:
            frame_capture = FrameCapture(frame_capture_config, frame_queue)
            frame_capture.run()
        except Exception as e:
            self.logger.critical(f"Failed to start frame capture process: {e}")
            self.logger.exception(e)
            self.logger.critical(f"Killing main process (pid = {os.getppid()})")
            os.kill(os.getppid(), signal.SIGTERM)
