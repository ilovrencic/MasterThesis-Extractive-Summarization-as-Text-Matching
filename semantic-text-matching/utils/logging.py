import logging

logger = logging.getLogger()

"""
Method that initializes the logger.
Logger is used for logging significant steps during the model training.

"""

def init_logger(log_file=None):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    reset_log(log_file)

    return logger


def reset_log(log_file=None):
    if log_file:
        f = open(log_file, "r+")
        f.truncate(0)
        f.close()
