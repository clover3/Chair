import os
import time

from job_manager.job_runner_with_server import JobRunnerF
from misc_lib import path_join


def work_fn(job_no):
    time.sleep(10)


def main():
    runner = JobRunnerF("/tmp", 11, "lock_test", work_fn)
    runner.start()



if __name__ == "__main__":
    main()