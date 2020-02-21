import shutil
import sys

from taskman_client.task_executer import get_last_mark, get_log_path, get_next_sh_path, get_sh_path_for_job_id


def check_if_exception(job_id):
    f = open(get_log_path(job_id), "r")
    last_lines = f.readlines()[-100:]
    for line in last_lines:
        if is_line_exception_like(line):
            return True
    return False


def is_line_exception_like(line):
    exception_keywords = ["RuntimeError", "Reporting Exception"]
    for keyword in exception_keywords:
        if keyword in line:
            return True
    return False


def get_exception_line(job_id):
    f = open(get_log_path(job_id), "r")
    last_lines = f.readlines()[-100:]
    for i, line in enumerate(last_lines):
        if is_line_exception_like(line):
            return "\n".join(last_lines[i-2:i+3])
    return ""


def rerun_from(start_job_no, end_job_no=100000):
    last_mark = get_last_mark()
    end_job_no = min(last_mark, end_job_no)

    for i in range(start_job_no, end_job_no):
        if check_if_exception(i):
            print(i)
#            print(get_exception_line(i))
            reschedule(i)


def reschedule(old_job_no):
    shutil.copyfile(get_sh_path_for_job_id(old_job_no), get_next_sh_path())


if __name__ == "__main__":
    if len(sys.argv) == 2:
        rerun_from(int(sys.argv[1]))
    elif len(sys.argv) == 3:
        rerun_from(int(sys.argv[1]), int(sys.argv[2]))