import logging
import sys

tray_logger = None


def setup_logging():
    global tray_logger
    if tray_logger is not None:
        return
    tray_logger = logging.getLogger("Tray")
    tray_logger.setLevel(logging.INFO)

    format_str = '%(levelname)s %(name)s %(asctime)s %(message)s'
    formatter = logging.Formatter(format_str,
                                  datefmt='%m-%d %H:%M:%S',
                                  )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    tray_logger.addHandler(ch)
    tray_logger.info("Application started")


setup_logging()
