import os
import sys
import time

from arg.robust.runner.collect_scores import make_ranked_list_from_multiple_files
from misc_lib import exist_or_mkdir, tprint
from taskman_client.add_job import run_job
from taskman_client.task_executer import check_job_mark


def add_jobs(sh_format_path , model_sub_path, save_dir, run_name):

    save_path_list = []
    exist_or_mkdir(save_dir)
    job_id_list = []
    for i in range(200, 250):
        save_path = os.path.join(save_dir, str(i))
        d = {
            "$model_subpath": model_sub_path,
            "$run_name": run_name,
            "$i": str(i),
            "$save_path": save_path
        }
        job_id = run_job(sh_format_path, d)
        job_id_list.append(job_id)
        save_path_list.append(save_path)

    return save_path_list, job_id_list


def wait_files(file_list, job_id_list):
    n_found = 0
    all_started = False
    all_started_time = -1
    while n_found == len(file_list):
        n_found = 0
        for file_path in file_list:
            if os.path.exists(file_path):
                n_found += 1
        time.sleep(60)

        if all_started:
            time_since_all_started = time.time() - all_started_time
            if time_since_all_started > 60 * 20:
                print("All job started but only {} found".format(n_found))
                print("Terminating")
                break
        else:
            n_job_started = 0
            for job_id in job_id_list:
                if check_job_mark(job_id):
                    n_job_started += 1

            if n_job_started == n_found:
                all_started = True
                all_started_time = time.time()

    time.sleep(360)


def main():
    model_name = sys.argv[1]
    step = int(sys.argv[2])
    model_sub_path = "{}/model.ckpt-{}".format(model_name, step)
    run_name = "{}_{}".format(model_name, step)
    save_root = "/mnt/disks/disk500/robust_score"
    sh_format_path = "robust_qck5/predict_template.sh"
    save_dir = os.path.join(save_root, "{}.score".format(run_name))
    tprint("Adding jobs..")
    save_path_list, job_id_list = add_jobs(sh_format_path, model_sub_path, save_dir, run_name)
    tprint("Waiting files")
    wait_files(save_path_list, job_id_list)
    tprint("Make ranked list")
    make_ranked_list_from_multiple_files(run_name, save_dir, 200, 250)


if __name__ == "__main__":
    main()