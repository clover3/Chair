import csv
from typing import List, Tuple

from misc_lib import average
from tab_print import tab_print
from tlm.qtype.partial_relevance.eval_score_dp_helper import get_eval_score_save_path


def load_eval_result(run_name) -> List[Tuple[str, float]]:
    save_path = get_eval_score_save_path(run_name)
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


def print_avg(run_name):
    avg, n_total, n_valid = calc_count_avg(run_name)
    tab_print("n_total", "n_valid", "avg")
    tab_print(n_total, n_valid, avg)


def calc_count_avg(run_name):
    eval_res = load_eval_result(run_name)
    n_total = len(eval_res)
    scores: List[float] = [t[1] for t in eval_res if t[1] is not None]
    n_valid = len(scores)
    avg = average(scores)
    return avg, n_total, n_valid