import csv
import os
from typing import List, Tuple

from cpath import output_path
from misc_lib import exist_or_mkdir


def get_eval_score_save_path_r(run_name):
    save_path = os.path.join(output_path, "qtype", "related_eval_score", "{}.score".format(run_name))
    return save_path


def save_eval_result_r(eval_res: List[Tuple[str, float]], run_name):
    save_path = get_eval_score_save_path_r(run_name)
    with open(save_path, "w", newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(eval_res)


def get_eval_score_save_path_b(run_name):
    dir_path = os.path.join(output_path, "qtype", "binary_related_eval_score")
    exist_or_mkdir(dir_path)
    save_path = os.path.join(dir_path, "{}.score".format(run_name))
    return save_path


def save_eval_result_b(eval_res: List[Tuple[str, float]], run_name):
    save_path = get_eval_score_save_path_b(run_name)
    with open(save_path, "w", newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(eval_res)

