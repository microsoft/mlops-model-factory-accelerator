import os

from src.frame_capture.config import FrameCaptureConfig
from src.frame_processor.config import EdgeInferenceConfig


class ConfigHandler:
    """Class for getting all configurations for the module and creating config objects"""

    def __init__(self, config_source: str = "environment") -> None:
        """
        Initialize ConfigHandler.

        @param:
            config_source (str): Source of all configurations. Defaulted to 'environment'
        """
        self.config_source = config_source

    def _get_config(self, config_name: str) -> str:
        """
        Get value of a specific configuration parameter from source.

        @param:
            config_name (str): Name of the configuration parameter to get
        @return:
            config_value (str): Value of the configuration parameter
        """
        if self.config_source == "environment":
            return self._read_from_env_variable(config_name)
        else:
            raise ValueError(f"Configuration source {self.config_source} is not supported")

    def _read_from_env_variable(self, config_name: str) -> str:
        """
        Read configuration from environment variable.

        @param:
            config_name (str): Name of the configuration parameter to get
        @return:
            config_value (str): Value of the configuration parameter
        """
        config_value = os.environ.get(config_name)
        if config_value is None:
            raise ValueError(f"Configuration {config_name} is not set in environment variables")
        return config_value

    def get_frame_capture_config(self, video_source=None, frame_rate=None) -> FrameCaptureConfig:
        """
        Get frame capture configuration.

        @return:
            FrameCaptureConfig (object): FrameCaptureConfig class object
        """
        return FrameCaptureConfig(
            video_source=self._get_config("CAMERA_PATH") if video_source is None else video_source,
            frame_rate=int(self._get_config("FRAME_RATE")) if frame_rate is None else frame_rate
        )

    def get_edge_inference_config(self, model_endpoint=None) -> EdgeInferenceConfig:
        """
        Get edge inference configuration.

        @return:
            EdgeInferenceConfig (object): EdgeInferenceConfig class object
        """
        return EdgeInferenceConfig(
            model_endpoint=self._get_config("MODEL_ENDPOINT") if model_endpoint is None else model_endpoint
        )
