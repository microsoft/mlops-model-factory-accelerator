class EdgeInferenceConfig:
    """
    Configuration for edge inferencing
    """

    def __init__(self, model_endpoint: str) -> None:
        """
        Initialize EdgeInferenceConfig.

        @param:
            model_endpoint (str): url of the ML module hosting the inference model
        """
        self.model_endpoint = model_endpoint
