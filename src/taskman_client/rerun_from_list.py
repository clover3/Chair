import sys

from taskman_client.rerun_failed import reschedule


def rerun_from_list(file_path):
    f = open(file_path)
    for line in f:
        job_no = int(line)
        print(job_no)
        reschedule(job_no)


if __name__ == "__main__":
    rerun_from_list(sys.argv[1])