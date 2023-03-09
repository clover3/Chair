import logging.config
import os
import re
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
    root_logger = logging.getLogger()
    root_logger.addHandler(ch)
    c_log.info("Chair logging init")
    tf_logger = logging.getLogger('tensorflow')
    tf_logger.propagate = False


class IgnoreFilter(logging.Filter):
    def __init__(self, targets):
        self.targets = targets
        super(IgnoreFilter).__init__()

    def filter(self, record):
        for pattern in self.targets:
            if pattern in record.msg:
                return False
        return True


class IgnoreFilterRE(logging.Filter):
    def __init__(self, re_list):
        self.re_list = re_list
        super(IgnoreFilterRE).__init__()

    def filter(self, record):
        for re_pattern in self.re_list:
            if re.search(re_pattern, record.msg) is not None:
                return False
        return True
