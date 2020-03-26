import os
import sys

import tensorflow as tf

from taskman_client.task_proxy import get_task_proxy


def main(_):
    if 'uuid' in os.environ:
        uuid_var = os.environ['uuid']
    else:
        uuid_var = None
    print("UUID:", uuid_var)
    ## end
    run_name = sys.argv[1]
    proxy = get_task_proxy(None, uuid_var)
    proxy.task_complete(run_name, "")


if __name__ == "__main__":
    tf.compat.v1.app.run()