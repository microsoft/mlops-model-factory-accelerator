class FrameCaptureConfig:
    """Configuration for frame capture."""

    def __init__(self, video_source: str, frame_rate: int) -> None:
        """
        Initialize FrameCaptureConfig.

        @param:
            video_source (str): Path to video source, E.g. camera path, RTSP URL, local video file etc.
            frame_rate (int): Frame rate at which video must be streamed

        """
        self.video_source = video_source
        self.frame_rate = frame_rate
