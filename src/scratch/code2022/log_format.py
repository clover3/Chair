import logging

from trainer_v2.chair_logging import c_log


def main():
    logger = c_log
    logger.setLevel(logging.INFO)
    s = "interesting problem"
    logger.info("Houston, we have a %s", s)
    logger.info("Houston, we have a %s")
    return NotImplemented


if __name__ == "__main__":
    main()