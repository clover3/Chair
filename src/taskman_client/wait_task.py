import os
import sys
import time

from taskman_client.task_proxy import get_task_manager_proxy

STATE_NOT_STARTED = 0
STATE_STARTED = 1
STATE_DONE = 2


def main():
    run_name = sys.argv[1]
    print("Waiting for task: {}".format(run_name))
    proxy = get_task_manager_proxy()
    info = proxy.query_task_status(run_name)
    state = STATE_NOT_STARTED
    # {'server_error': False, 'started': True, 'terminated': True, 'task_error': False}
    while not info['terminated']:
        if state == STATE_NOT_STARTED and info['started']:
            state = STATE_STARTED
            print("Task started")

        time.sleep(10)
        info = proxy.query_task_status(run_name)

    print(info)
    print("Task {} is terminated".format(run_name))

    if info['task_error']:
        print("process terminated with error")


if __name__ == "__main__":
    main()