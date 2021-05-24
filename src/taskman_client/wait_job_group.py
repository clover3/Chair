import os
import sys
import time

from taskman_client.task_proxy import get_task_manager_proxy

STATE_PENDING = 0
STATE_STARTED = 1
STATE_DONE = 2


def main():
    job_name = sys.argv[1]
    proxy = get_task_manager_proxy()
    info = proxy.query_job_group_status(job_name)
    keys = ['started', 'num_done', 'max_job']
    last_key_val = {}

    while info['state'] != STATE_DONE:
        any_change = False
        for key in keys:
            if key not in last_key_val or last_key_val[key] != info[key]:
                any_change = True
            last_key_val[key] = info[key]

        if any_change:
            print("started/done/max={}/{}/{}".
                  format(info['started'], info['num_done'], info['max_job']))

        time.sleep(10)
        info = proxy.query_job_group_status(job_name)

    print("JOB {} is done".format(job_name))


if __name__ == "__main__":
    main()