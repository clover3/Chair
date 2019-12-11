import time

import tensorflow as tf

from taskman_client.task_proxy import get_task_proxy


def task():
    time.sleep(60)

def main(_):
    task_proxy = get_task_proxy()
    run_name = "dummy_run"
    task_proxy.task_start(run_name)
    task()
    task_proxy.task_complete(run_name)


if __name__ == "__main__":
    tf.compat.v1.app.run()
