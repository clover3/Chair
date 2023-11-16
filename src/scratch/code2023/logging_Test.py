import logging
import sys


class SomeClass:
    def __init__(self):
        self.tray_logger = logging.getLogger("tray")
        format_str = '%(levelname)s\t%(name)s \t%(asctime)s %(message)s'
        formatter = logging.Formatter(format_str,
                                      datefmt='%m-%d %H:%M:%S',
                                      )
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        self.tray_logger.addHandler(ch)
        self.tray_logger.setLevel(logging.INFO)
        self.tray_logger.info("Application started")


def main():
    SomeClass()
    print("Done")
    return NotImplemented


if __name__ == "__main__":
    main()