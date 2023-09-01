"""A module for running smoke tests."""

import importlib

from model_factory.common.smoke_test.modules.TestModule.response import Response

class TestRunner:
    """A class for executing smoke tests."""

    def execute_smoke_test(self, port, test_module) -> Response:
        """Execute a smoke test for the specified module.

        Args:
            test_module (str): The name of the module containing the smoke test.

        Returns:
            MethodResponse: A response object indicating the result of the smoke test.
        """
        try:
            test_module = importlib.import_module(f"model_factory.{test_module}.smoke_test")
            smoke_test_instance = test_module.get_instance(port)
            return smoke_test_instance.execute_tests()
        except ModuleNotFoundError:
            print(f"Module {test_module} not found")
            return Response(500, {"status": "FAIL"})