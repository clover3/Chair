import os

import tensorflow as tf

from taskman_client.task_proxy import get_task_proxy
from taskman_client.wrapper import flag_to_run_name
from tlm.benchmark.report import get_hp_str_from_flag
from tlm.training.train_flags import *


def main(_):
    if 'uuid' in os.environ:
        uuid_var = os.environ['uuid']
    else:
        uuid_var = None
    run_name = flag_to_run_name(FLAGS)
    flags_str = get_hp_str_from_flag(FLAGS)
    proxy = get_task_proxy(FLAGS.tpu_name, uuid_var)
    proxy.task_start(run_name, flags_str)

if __name__ == "__main__":
    tf.compat.v1.app.run()