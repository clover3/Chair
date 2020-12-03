import json
import os
import sys

from arg.robust.eval_helper import add_jobs, wait_files
from arg.robust.runner.collect_scores import make_ranked_list_from_multiple_files
from cpath import output_path
from misc_lib import tprint, exist_or_mkdir


def save_runs_log(info, name):
    log_dir = os.path.join(output_path, "runs_log")
    exist_or_mkdir(log_dir)
    save_path = os.path.join(log_dir, name + ".json")
    json.dump(info, open(save_path, "w"), indent=True)


def get_save_dir(job_group_name):
    save_root = "/mnt/disks/disk500/robust_score"
    save_dir = os.path.join(save_root, "{}.score".format(job_group_name))
    return save_dir

def main():
    model_name = sys.argv[1]
    step = int(sys.argv[2])
    model_sub_path = "{}/model.ckpt-{}".format(model_name, step)
    job_group_name = "{}_{}".format(model_name, step)

    sh_format_path = sys.argv[3]
    #sh_format_path = "robust_qck6/predict_template.sh"
    save_dir = get_save_dir(job_group_name)
    data_st = 200
    data_ed = 250

    tprint("Adding jobs..")
    job_info_list = add_jobs(sh_format_path, model_sub_path, save_dir,
                             job_group_name, data_st, data_ed)
    runs_info = {
        'job_group_name': job_group_name,
        'save_dir': save_dir,
        'job_info_list': job_info_list,
        'data_st': data_st,
        'data_ed': data_ed,
        'sh_format_path': sh_format_path,
        'model_sub_path': model_sub_path
    }
    runs_name = "{}_{}".format(model_name, step)
    save_runs_log(runs_info, runs_name)

    tprint("Waiting files")
    wait_files(job_info_list)

    tprint("Make ranked list")
    make_ranked_list_from_multiple_files(job_group_name, save_dir, data_st, data_ed)


if __name__ == "__main__":
    main()
