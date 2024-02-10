import logging
import sys

c2_log = None
if c2_log is None:
    c2_log = logging.getLogger('C2')
    c2_log.setLevel(logging.DEBUG)
    format_str = '%(levelname)s\t%(name)s \t%(asctime)s %(message)s'
    formatter = logging.Formatter(format_str,
                                  datefmt='%m-%d %H:%M:%S',
                                  )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(ch)
    c2_log.info("C2 logging init")
