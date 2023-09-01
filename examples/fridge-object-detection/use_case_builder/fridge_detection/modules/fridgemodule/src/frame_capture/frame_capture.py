import cv2
import logging
import os
import time

from src.common import constants
from src.frame_capture.config import FrameCaptureConfig
from src.multiprocessing.queue import ProcessQueue


class FrameCapture:
    """Class for capturing frames from a video source.
    """

    def __init__(self, frame_capture_config: FrameCaptureConfig, queue: ProcessQueue) -> None:
        """
        Initialize FrameCapture.
        """
        self.logger = logging.getLogger(__name__)

        self.queue = queue
        self.frame_capture_config = frame_capture_config

        self.video = None

    def run(self) -> None:
        """
        Start frame capture from video source
        """
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        while True:
            self.logger.info("Connecting to video source...")
            self.video = cv2.VideoCapture(self.frame_capture_config.video_source, cv2.CAP_FFMPEG)
            if not self.video.isOpened():
                self.logger.error(
                    f"Unable to capture video stream from {self.frame_capture_config.video_source}, "
                    "attempting to reconnect to video source..."
                )
                time.sleep(constants.CAMERA_RECONNECT_WAIT_TIME_SECS)
                continue
            self.logger.info("Video stream captured successfully")

            self._capture_frame(self.frame_capture_config.frame_rate)

    def _get_skip_frames(self, desired_frame_rate):
        """
        Calculate number of frames to skip to achieve desired frame rate.
        """

        video_frame_rate = self.video.get(cv2.CAP_PROP_FPS)
        skip_frames = video_frame_rate/desired_frame_rate
        return skip_frames

    def _capture_frame(self, desired_frame_rate):
        """
        Capture frame from video source as per the desired frame rate.
        """

        skip_frames = self._get_skip_frames(desired_frame_rate)

        frame_nbr = 0
        success, frame = self.video.read()
        while success:
            if frame_nbr % skip_frames == 0:
                self.queue.add_item(frame)
            success, frame = self.video.read()
            frame_nbr += 1
        self.logger.info("Reached end of video stream, restarting video...")
        self.video.release()
