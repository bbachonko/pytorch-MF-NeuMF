import logging


def setup_logger(name: str) -> logging.Logger:
    """
    Sets up and returns a basic logger with INFO level and no handlers.

    Parameters:
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.get(name)
    logger.setLevel(logging.INFO)
    return logger
