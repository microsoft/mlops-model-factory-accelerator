"""A module defining the BaseSmokeTestModule class."""
import requests
import json

from json_schema import json_schema
from abc import ABC, abstractmethod
from typing import Callable

from model_factory.common.smoke_test.modules.TestModule.response import Response

def default_invocation(port, request_json):
    """Invoke the smoke test and return the results.

    Args:
        port (int): The port number to use for the smoke test.
        request_json (str): The JSON payload to send to the smoke test.

    Returns:
        str: The response of the model as a string.
    """
    url = "http://127.0.0.1:{}/score".format(port)
    response = requests.post(
                        url=url, data=request_json, 
                        headers={"Content-Type": "application/json"}
                    )
    if response.status_code != 200:
        raise Exception("Received bad response from the model: " + response)
    
    print(f"Smoke Test: received response from model{response.text}")
    return response.text

def default_json_comparision(expected_result, actual_result) -> Response:
    """Validate the results of the smoke test.

    Args:
        expected_result (Any): The expected results of the smoke test.
        actual_result (str): The actual results of the smoke test.

    Returns:
        Response: A response indicating whether the results are matching or not.
    """
    expected_result_schema = json_schema.dumps(expected_result)

    result = json_schema.match(actual_result, expected_result_schema)

    if result:
        return Response(200, "Smoke Test: all params are equal")
    else:
        return Response(400, "Smoke Test: mismtach in params ")


class BaseSmokeTestModule(ABC):
    """A base class for smoke test modules."""

    def __init__(self,
                 invoker: Callable[[int, str], str] = default_invocation,
                 validator: Callable[[str, str], Response] = default_json_comparision):
        """Initialize a new instance of the BaseSmokeTestModule class.

        Args:
            invoker (Callable[[int, str], str]): A function to invoke the smoke test. Defaults to default_invocation.
            validator (Callable[[str, str], Response]): A function to validate the results of the smoke test. Defaults to default_json_validation.
        """
        self.invoker = invoker
        self.validator = validator

    def run(self, port, request_json, expected_result):
        """Execute a single smoke test for the given request and asserts with the expected result."""
        actual_result = self.invoker(port, request_json)
        return self.validator(expected_result, actual_result)

    @abstractmethod
    def execute_tests(self):
        """Execute the smoke tests."""
        pass