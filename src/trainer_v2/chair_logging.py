import logging.config
import sys

# logging.config.fileConfig('logging.conf')

c_log = None
if c_log is None:
    c_log = logging.getLogger('chair')
    c_log.setLevel(logging.INFO)
    format_str = '%(levelname)s\t%(name)s \t%(asctime)s %(message)s'
    formatter = logging.Formatter(format_str,
                                  datefmt='%m-%d %H:%M:%S',
                                  )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)

    c_log.addHandler(ch)
    c_log.info("Chair logging init")