import csv
import os
from typing import List, Tuple, Optional

from cache import save_list_to_jsonl, load_list_from_jsonl
from cpath import output_path
from misc_lib import exist_or_mkdir
from alignment.data_structure.eval_data_structure import PerProblemEvalResult


def get_eval_score_save_path_r(run_name):
    save_path = os.path.join(output_path, "qtype", "related_eval_score", "{}.score".format(run_name))
    return save_path


def save_eval_result_r(eval_res: List[Tuple[str, float]], run_name):
    save_path = get_eval_score_save_path_r(run_name)
    with open(save_path, "w", newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(eval_res)


def get_eval_score_save_path_b_single(run_name):
    dir_path = os.path.join(output_path, "qtype", "binary_related_eval_score")
    exist_or_mkdir(dir_path)
    save_path = os.path.join(dir_path, "{}.score".format(run_name))
    return save_path


def get_eval_score_save_path_b_all(run_name):
    dir_path = os.path.join(output_path, "qtype", "binary_related_eval_score_all")
    exist_or_mkdir(dir_path)
    save_path = os.path.join(dir_path, "{}.score".format(run_name))
    return save_path


def save_eval_result_b_single(eval_res: List[Tuple[str, float]], run_name):
    save_path = get_eval_score_save_path_b_single(run_name)
    write_tsv(eval_res, save_path)


def write_tsv(eval_res, save_path):
    with open(save_path, "w", newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(eval_res)


def save_eval_result_b(eval_res: List[PerProblemEvalResult], run_name):
    save_path = get_eval_score_save_path_b_all(run_name)
    save_list_to_jsonl(eval_res, save_path)


def load_eval_result_r(run_name) -> List[Tuple[str, float]]:
    save_path = get_eval_score_save_path_r(run_name)
    return load_eval_result_from_path(save_path)


def load_eval_result_b_single(run_name) -> List[Tuple[str, float]]:
    save_path = get_eval_score_save_path_b_single(run_name)
    return load_eval_result_from_path(save_path)


def load_eval_result_from_path(save_path):
    with open(save_path, "r") as f:
        reader = csv.reader(f, delimiter='\t')

        def parse_row(row):
            problem_id = row[0]
            try:
                score = float(row[1])
            except ValueError:
                score = None
            return problem_id, score

        return list(map(parse_row, reader))


def load_eval_result_b_all(run_name) -> List[Tuple[str, List[Optional[float]]]]:
    save_path = get_eval_score_save_path_b_all(run_name)
    return load_list_from_jsonl(save_path, PerProblemEvalResult.from_json)


def get_run_name(dataset, method, policy_name):
    run_name = "{}_{}_{}".format(dataset, method, policy_name)
    return run_name


def is_run_exist(dataset, method, policy_name):
    save_path = get_eval_score_save_path_b_all(get_run_name(dataset, method, policy_name))
    return os.path.exists(save_path)
