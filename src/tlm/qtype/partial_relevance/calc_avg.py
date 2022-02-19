from typing import List

from misc_lib import average
from tab_print import tab_print
from tlm.qtype.partial_relevance.eval_score_dp_helper import load_eval_result_r


def print_avg(run_name):
    eval_res = load_eval_result_r(run_name)
    avg, n_total, n_valid = calc_count_avg(eval_res)
    tab_print("n_total", "n_valid", "avg")
    tab_print(n_total, n_valid, avg)


def calc_count_avg(eval_res):
    n_total = len(eval_res)
    scores: List[float] = [t[1] for t in eval_res if t[1] is not None]
    n_valid = len(scores)
    avg = average(scores)
    return avg, n_total, n_valid