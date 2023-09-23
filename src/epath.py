import os

job_man_dir = os.environ["JOB_MAN"]


def at_job_man_dir1(folder_name):
    return os.path.join(job_man_dir, folder_name)


def at_job_man_dir2(folder_name, file_name):
    return os.path.join(job_man_dir, folder_name, file_name)


