import sys

from arg.robust.eval_helper import add_jobs, wait_files
from arg.robust.runner.collect_scores import make_ranked_list_from_multiple_files
from arg.robust.runner.run_eval_jobs import get_save_dir, save_run_group_info
from misc_lib import tprint


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
    job_list = [205,206,207,210,211,216,220,225,227,228,230,231,238,240,241,242,243,244,245,246,248]
    tprint("Adding jobs..")
    job_info_list = add_jobs(sh_format_path, model_sub_path, save_dir,
                             job_group_name, job_list)
    run_group_info = {
        'job_group_name': job_group_name,
        'save_dir': save_dir,
        'job_info_list': job_info_list,
        'data_st': data_st,
        'data_ed': data_ed,
        'sh_format_path': sh_format_path,
        'model_sub_path': model_sub_path,
        'rerun_jobs': False,
    }
    runs_name = "{}_{}".format(model_name, step)
    save_run_group_info(run_group_info, runs_name)

    tprint("Waiting files")
    wait_files(job_info_list)

    tprint("Make ranked list")
    make_ranked_list_from_multiple_files(job_group_name, save_dir, data_st, data_ed)


if __name__ == "__main__":
    main()
