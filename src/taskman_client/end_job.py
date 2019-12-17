import os

import tensorflow as tf

from taskman_client.task_proxy import get_task_proxy
from taskman_client.wrapper import flag_to_run_name
from tlm.training.train_flags import *


def main(_):
    if 'uuid' in os.environ:
        uuid_var = os.environ['uuid']
    else:
        uuid_var = None
    run_name = flag_to_run_name(FLAGS)
    proxy = get_task_proxy(FLAGS.tpu_name, uuid_var)
    proxy.task_complete(run_name, "")


if __name__ == "__main__":
    tf.compat.v1.app.run()