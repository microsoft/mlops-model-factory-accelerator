import requests

from model_factory.common.smoke_test.modules.TestModule.response import Response

class Healthcheck:
    """
    A class for performing health checks on a service.

    Attributes:
        None

    Methods:
        execute(port: int) -> Response: Performs a health check on the service running on the specified port.
    """
    def execute(self, port: int) -> Response:
        """
        Performs a health check on the service running on the specified port.

        Args:
            port (int): The port number on which the service is running.

        Returns:
            Response: A Response object containing the status code and status message of the health check.
        """
        url = "http://127.0.0.1:{}/healthcheck".format(port)
        res = requests.get(url=url, timeout=5)
        if res.status_code == 200:
            return Response(200, {"status": "OK"})
        else:
            return Response(500, {"status": "FAIL"})