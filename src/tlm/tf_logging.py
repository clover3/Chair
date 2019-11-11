import logging as py_logging

logging = py_logging.getLogger('tensorflow')
logging.removeHandler(logging.handlers[0])
logging.info("1")
