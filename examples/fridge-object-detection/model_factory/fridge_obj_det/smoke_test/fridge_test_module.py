"""A module defining the FridgeTestModule class."""

import base64
import json

from model_factory.common.smoke_test.modules.TestModule.base_test_module import BaseSmokeTestModule

fs_prefix = "/app/model_factory/fridge_obj_det/smoke_test"
class FridgeTestModule(BaseSmokeTestModule):
    """A class for testing the fridge model."""

    def __init__(self, port):
        """Initialize a new instance of the FridgeTestModule class.

        Args:
            port (int): The port number to use for the smoke test.
        """
        self.port = port
        super().__init__()

    def execute_tests(self):
        """Execute the smoke tests.

        Returns:
            Response: A response indicating whether the tests passed or failed.
        """
        test_file_name = f"{fs_prefix}/test_data/test_sample.jpg"
        with open(test_file_name, "rb") as image_file:
            image_bytes = image_file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf8")
        request_payload = json.dumps({"images": [image_base64]})
        expected_result = open(f"{fs_prefix}/test_data/test_response.json", "r").read()
        return super().run(self.port,
                           request_json=request_payload,
                           expected_result=expected_result)
