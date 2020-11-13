import os
import sys
import time

from misc_lib import get_dir_files2
from taskman_client.task_proxy import get_task_proxy


def get_job_name_from_done_path(done_path):
    p = os.path.basename(done_path)
    if p.endswith("_done"):
        return p[:-5]
    return p


def main():
    job_done_dir = sys.argv[1]
    num_jobs = int(sys.argv[2])

    proxy = get_task_proxy(None, None)
    uuid_var = proxy.uuid_var
    run_name = get_job_name_from_done_path(job_done_dir)
    proxy.task_start(run_name)
    print(uuid_var)

    last_num_files = -1
    while last_num_files < num_jobs:
        files = get_dir_files2(job_done_dir)

        for file_path in files:
            # file name must be int
            num = int(os.path.basename(file_path))

        num_files = len(files)
        if num_files != last_num_files:
            print("# Job done: ", num_files)
            last_num_files = num_files
        time.sleep(10)

    print("All jobs done")
    proxy.task_complete(run_name)


if __name__ == "__main__":
    main()
