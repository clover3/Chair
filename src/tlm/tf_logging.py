import logging
from absl import logging as ab_logging
import time

logging.root.addHandler(logging.StreamHandler())
tf_logging = logging.getLogger('tensorflow')
tf_logging.info("1")


class MyFormatter(logging.Formatter):
    def prefix(self, record):
        """Returns the absl log prefix for the log record.

        Args:
        record: logging.LogRecord, the record to get prefix for.
        """
        created_tuple = time.localtime(record.created)
        created_microsecond = int(record.created % 1.0 * 1e6)

        critical_prefix = ''
        level = record.levelno

        return '%s %s [%02d:%02d:%02d %s:%d] %s' % (
            logging._levelToName[level],
            record.name,
          created_tuple.tm_hour,
          created_tuple.tm_min,
          created_tuple.tm_sec,
          record.filename,

         record.lineno,
          critical_prefix)

    def format(self, record):
        result = super().format(record)
        result = self.prefix(record) + result
        return result


class TFFilter(logging.Filter):
    excludes = ["Outfeed finished for iteration", "TPUPollingThread found TPU"]
    def filter(self, record):
        for e in self.excludes:
            if e in record.msg:
                return False
        return True


h = ab_logging.get_absl_handler()
h.setFormatter(MyFormatter())

logging.getLogger('oauth2client.transport').setLevel(logging.WARNING)
tf_logging.addFilter(TFFilter())
