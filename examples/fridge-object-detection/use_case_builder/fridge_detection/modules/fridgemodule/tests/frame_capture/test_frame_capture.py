import cv2
import numpy as np
import unittest
from unittest.mock import MagicMock

from src.frame_capture.config import FrameCaptureConfig
from src.frame_capture.frame_capture import FrameCapture
from src.multiprocessing.queue import ProcessQueue


class FrameCaptureTestCase(unittest.TestCase):

    def setUp(self, *args):
        self.frame_capture_config = FrameCaptureConfig("tests/frame_capture/test_video.mp4", 1)
        self.queue = ProcessQueue()
        self.frame_capture = FrameCapture(self.frame_capture_config, self.queue)

    def tearDown(self):
        self.frame_capture.video.release()
        self.queue.shared_queue.close()

    def test_get_skip_frames(self):
        # Arrange
        self.frame_capture.video = MagicMock()
        self.frame_capture.video.get.return_value = 30
        desired_frame_rate = 10

        # Act
        skip_frames = self.frame_capture._get_skip_frames(desired_frame_rate)

        # Assert
        self.assertEqual(skip_frames, 3.0)

    def test_capture_frame_adds_to_queue(self):
        # Arrange
        self.frame_capture.video = cv2.VideoCapture("tests/frame_capture/test_video.mp4", cv2.CAP_FFMPEG)
        self.frame_capture._get_skip_frames = MagicMock(return_value=30)
        desired_frame_rate = 1
        self.frame_capture.queue.add_item = MagicMock()

        # Act
        self.frame_capture._capture_frame(desired_frame_rate)

        # Assert
        self.assertEqual(self.frame_capture.queue.add_item.call_count, 10)

    def test_capture_frame_queue_item_is_frame(self):
        # Arrange
        self.frame_capture.video = cv2.VideoCapture("tests/frame_capture/test_video.mp4", cv2.CAP_FFMPEG)
        self.frame_capture._get_skip_frames = MagicMock(return_value=30)
        desired_frame_rate = 1

        # Act
        self.frame_capture._capture_frame(desired_frame_rate)

        # Assert
        frame = self.frame_capture.queue.get_item()
        self.assertIsInstance(frame, np.ndarray)

    def test_capture_frame_when_frame_is_none(self):
        # Arrange
        self.frame_capture.video = MagicMock()
        self.frame_capture.video.get.return_value = 30
        self.frame_capture.video.read.return_value = False, None
        desired_frame_rate = 1

        # Act
        self.frame_capture._capture_frame(desired_frame_rate)

        # Assert
        self.assertEqual(self.queue.shared_queue.qsize(), 0)


if __name__ == '__main__':
    unittest.main()
