import json
import os
import sys

from arg.robust.eval_helper import wait_files
from arg.robust.runner.collect_scores import make_ranked_list_from_multiple_files
from misc_lib import tprint
from taskman_client.add_job import run_job


def rerun_jobs(sh_format_path, model_sub_path,
               save_dir, job_group_name, job_info_list):
    new_job_info_list = []
    for job_info in job_info_list:
        if not os.path.exists(job_info['save_path']):
            file_name = os.path.basename(job_info['save_path'])
            job_no = int(file_name)
            save_path = os.path.join(save_dir, str(job_no))
            run_name = "{}-{}".format(job_group_name, job_no)

            d = {
                "$model_subpath": model_sub_path,
                "$run_name": run_name,
                "$i": str(job_no),
                "$save_path": save_path
            }
            job_id = run_job(sh_format_path, d)
            new_job_info = {
                'job_id': job_id,
                'save_path': job_info['save_path']
            }
            new_job_info_list.append(new_job_info)
    return new_job_info_list


def main():
    info_path = sys.argv[1]
    # TODO remove
    # sh_format_path = sys.argv[2]
    # model_name = sys.argv[3]
    # step = int(sys.argv[4])
    # model_sub_path = "{}/model.ckpt-{}".format(model_name, step)
    # TODO remove end

    run_info = json.load(open(info_path, "r"))
    job_info_list = run_info['job_info_list']
    job_group_name = run_info['job_group_name']
    save_dir = run_info['save_dir']
    data_st = run_info['data_st']
    data_ed = run_info['data_ed']
    sh_format_path = run_info['sh_format_path']
    model_sub_path = run_info['model_sub_path']
    if 'rerun_jobs' in run_info and run_info['rerun_jobs']:
        new_job_info_list = rerun_jobs(sh_format_path, model_sub_path, save_dir, job_group_name,
               job_info_list)
        job_info_list = new_job_info_list

    print("len(job_info_list)", len(job_info_list))
    tprint("Waiting files")
    wait_files(job_info_list)
    tprint("Make ranked list")
    make_ranked_list_from_multiple_files(job_group_name, save_dir, data_st, data_ed)


if __name__ == "__main__":
    main()
