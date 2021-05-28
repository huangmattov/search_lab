import logging
import sys


class LoggingHelper(object):

    def __init__(self, name):
        self.name = name
        self.logger = self.create_logger()

    def create_logger(self, stream=sys.stdout):
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        handler = logging.StreamHandler(stream)
        handler.setFormatter(formatter)

        logger = logging.getLogger(self.name)
        if logger.hasHandlers():
            logger.handlers.clear()

        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        return logger


