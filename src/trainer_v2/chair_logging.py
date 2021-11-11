import logging.config
import sys

# logging.config.fileConfig('logging.conf')

c_log = None
if c_log is None:
    c_log = logging.getLogger('chair')
    c_log.info("Chair logging init")
c_log.setLevel(logging.INFO)
c_log.addHandler(logging.StreamHandler(sys.stdout))