import logging

logger = logging.Logger("myLogger")
logger.setLevel(logging.DEBUG)
c_handler = logging.StreamHandler()
logger.addHandler(c_handler)

class ContextFilter(logging.Filter):
    def filter(self, record):
        if "GOOD" in record.msg:
            return False
        return True

logger.addFilter(ContextFilter())

logger.debug('This is a debug message')
logger.debug('This is a BAD debug message')
logger.debug('This is a GOOD debug message')
logger.info('This is an info message')
logging.warning('This is a warning message')
logging.error('This is an error message')
logging.critical('This is a critical message')
