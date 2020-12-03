import os

from taskman_client.task_executer import get_last_mark, get_new_job_id, get_sh_path_for_job_id


def main():
    last_mask = get_last_mark()
    job_ed = get_new_job_id()

    for i in range(last_mask+1, job_ed):
        sh_path = get_sh_path_for_job_id(i)
        os.remove(sh_path)

    print("Removed job {}-{}".format(last_mask+1, job_ed))


if __name__ == "__main__":
    main()

