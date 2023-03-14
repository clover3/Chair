import logging

from trainer_v2.chair_logging import IgnoreFilterRE


def ignore_distilbert_save_warning():
    ignore_msg = [
        r"Skipping full serialization of Keras layer .*Dropout",
        r"Found untraced functions such as"
    ]
    ignore_filter = IgnoreFilterRE(ignore_msg)
    tf_logging = logging.getLogger("tensorflow")
    tf_logging.addFilter(ignore_filter)