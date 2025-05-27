import logging
import sys

def setup_logger(name: str) -> logging.Logger:
    """
    Sets up and returns a basic logger with INFO level and StreamHandler if no handlers exist.

    Parameters:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.propagate = False  # optional: avoid double logging if root also prints

    return logger
