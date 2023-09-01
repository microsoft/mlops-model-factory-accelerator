
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from src.common import constants


def retry_session():
    """ Retry session for requests """
    retry_strategy = Retry(
        total=constants.REQUEST_MAX_RETRIES,  # Maximum number of retries
        backoff_factor=constants.REQUEST_BACKOFF_FACTOR,  # Exponential backoff factor (2 seconds, 4 seconds, 8 seconds, etc.)
        status_forcelist=[500, 502, 503, 504],  # Retry on specific HTTP status codes
        allowed_methods=["POST"]  # Retry only on POST requests
    )

    # Create a session with the retry strategy
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
