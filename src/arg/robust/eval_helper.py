import os
import time
from typing import List, Dict

from misc_lib import exist_or_mkdir, tprint
from taskman_client.add_job import run_job
from taskman_client.task_executer import check_job_mark


def add_jobs(sh_format_path, model_sub_path, save_dir, job_group_name, data_st, data_ed):
    save_path_list = []
    exist_or_mkdir(save_dir)
    job_id_list = []
    job_info_list: List[Dict] = []
    for i in range(data_st, data_ed):
        save_path = os.path.join(save_dir, str(i))
        run_name = "{}-{}".format(job_group_name, i)
        d = {
            "$model_subpath": model_sub_path,
            "$run_name": run_name,
            "$i": str(i),
            "$save_path": save_path
        }
        job_id = run_job(sh_format_path, d)
        job_id_list.append(job_id)
        save_path_list.append(save_path)
        job_info = {
            'job_id': job_id,
            'save_path': save_path,
            'data_no': i,
        }
        job_info_list.append(job_info)

    return job_info_list


def wait_files(job_info_list):
    num_files = len(job_info_list)
    file_list = list([info['save_path'] for info in job_info_list])
    job_id_list = list([info['job_id'] for info in job_info_list])
    n_found = 0
    all_started = False
    all_started_time = -1
    last_new_file = -1
    last_found = 0
    while n_found < num_files:
        n_found = 0
        for file_path in file_list:
            if os.path.exists(file_path):
                n_found += 1
        time.sleep(60)

        if all_started:
            time_since_all_started = time.time() - all_started_time
            time_since_last_new_file = time.time() - last_new_file
            if time_since_all_started > 60 * 20 and time_since_last_new_file > 60 * 10:
                tprint("All job started but only {} found".format(n_found))
                tprint("Terminating")
                break
        else:
            n_job_started = 0
            for job_id in job_id_list:
                if check_job_mark(job_id):
                    n_job_started += 1

            if n_job_started == n_found:
                tprint("all jobs started")
                all_started = True
                all_started_time = time.time()

        if n_found > last_found:
            tprint("Num files : {}".format(n_found))
            last_new_file = time.time()
        last_found = n_found
    time.sleep(360)