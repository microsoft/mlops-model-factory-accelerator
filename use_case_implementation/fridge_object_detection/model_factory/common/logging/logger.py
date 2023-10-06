"""Reusable logger for model_factory."""
import logging
import sys


def get_logger(name: str = "dataops", level: int = logging.DEBUG) -> logging.Logger:
    """Get logger for dataops module.

    Args:
        name (str, optional): Logger name. Defaults to "dataops".
        level (int, optional): Log level. Defaults to logging.DEBUG.

    Returns:
        logging.Logger: named logger.
    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(handler)

    return logger
