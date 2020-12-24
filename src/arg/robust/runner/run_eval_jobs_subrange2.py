import json
import sys
from typing import List

from arg.qck.multi_file_save_to_trec_form import save_over_multiple_files
from arg.robust.eval_helper import add_jobs, wait_files
from arg.robust.runner.collect_scores import make_ranked_list_from_multiple_files
from arg.robust.runner.run_eval_jobs import get_save_dir, save_run_group_info
from list_lib import lmap
from misc_lib import tprint


def main():
    model_name = sys.argv[1]
    step = int(sys.argv[2])
    model_sub_path = "{}/model.ckpt-{}".format(model_name, step)
    job_group_name = "{}_{}".format(model_name, step)
    sh_format_path = sys.argv[3]
    template_json = sys.argv[4]
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

    save_config = json.load(open(template_json, "r"))
    pred_file_list: List[str] = lmap(lambda d: d['save_path'], job_info_list)
    tprint("Make ranked list")
    make_ranked_list_from_multiple_files(job_group_name, save_dir, data_st, data_ed)
    save_over_multiple_files(pred_file_list,
                             save_config['info_path'],
                             runs_name,
                             save_config["input_type"],
                             save_config['max_entry'],
                             save_config['combine_strategy'],
                             save_config['score_type'],
                             )


if __name__ == "__main__":
    main()
