"""This module contains the Response class."""

class Response:
    """A class representing an HTTP response.

    Attributes:
        status (int): The HTTP status code.
        payload (str): The response payload.
    """

    def __init__(self, status: int, payload: str) -> None:
        """Initializes a new instance of the Response class.

        Args:
            status (int): The HTTP status code.
            payload (str): The response payload.
        """
        self.status = status
        self.payload = payload
