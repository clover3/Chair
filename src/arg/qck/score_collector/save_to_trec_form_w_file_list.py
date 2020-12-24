import json
from typing import List, Dict

from arg.qck.multi_file_save_to_trec_form import save_over_multiple_files
from exec_lib import run_func_with_config
from list_lib import lmap


def main(run_config):
    job_info_list: List[Dict] = run_config['job_info_list']
    pred_file_list: List[str] = lmap(lambda d: d['save_path'], job_info_list)
    save_config_path = run_config['save_config_path']
    save_config = json.load(open(save_config_path, "r"))

    save_over_multiple_files(pred_file_list,
                             save_config['info_path'],
                             run_config['job_group_name'],
                             save_config["input_type"],
                             save_config['max_entry'],
                             save_config['combine_strategy'],
                             save_config['score_type'],
                             )


if __name__ == "__main__":
    run_func_with_config(main)