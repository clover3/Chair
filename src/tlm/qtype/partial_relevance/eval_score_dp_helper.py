import csv
from typing import List, Iterable, Callable, Dict, Tuple, Set
from cpath import output_path
import os


def get_eval_score_save_path(run_name):
    save_path = os.path.join(output_path, "qtype", "related_eval_score", "{}.score".format(run_name))
    return save_path


def save_eval_result(eval_res: List[Tuple[str, float]], run_name):
    save_path = get_eval_score_save_path(run_name)
    with open(save_path, "w", newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(eval_res)
