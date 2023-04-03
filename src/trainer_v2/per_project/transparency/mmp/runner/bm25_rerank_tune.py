import random
import math

from adhoc.bm25_class import BM25
from dataset_specific.msmarco.passage.load_term_stats import load_msmarco_passage_term_stat
from misc_lib import get_second
from trainer_v2.per_project.transparency.mmp.eval_helper.eval_line_format import eval_dev100_for_tune, \
    predict_and_save_scores
from typing import List, Iterable, Callable, Dict, Tuple, Set
import numpy as np
from trainer_v2.per_project.transparency.mmp.eval_helper.hp_utils import generate_hyperparameters


def param_to_str(bm25_params: Dict[str, float]):
    return "_".join([f"{k}_{v}" for k, v in bm25_params.items()])


def get_bm25(bm25_params: Dict[str, float]) -> BM25:
    cdf, df = load_msmarco_passage_term_stat()
    bm25 = BM25(df, cdf, 25,
                bm25_params['k1'],
                bm25_params['k2'],
                bm25_params['b'],
                )
    return bm25


def run_eval_on_bm25(bm25_params) -> float:
    bm25 = get_bm25(bm25_params)
    run_name = "bm25_" + param_to_str(bm25_params)
    dataset = "dev_sample100"
    predict_and_save_scores(bm25.score, dataset, run_name)
    score = eval_dev100_for_tune(dataset, run_name)
    return score


def enum_125(st, ed, factor: float=1):
    cursor = st
    while cursor <= ed:
        yield cursor * factor
        if str(cursor)[0] == '1':
            cursor *= 2
        if str(cursor)[0] == '2':
            cursor = int(cursor / 2 * 5)
        if str(cursor)[0] == '5':
            cursor *= 2


def enum_10_exp(st, ed):
    for i in range(st, ed+1):
        yield math.pow(10, i)


def main():
    params_range = {
        'k1': [0.1],
        'k2': enum_10_exp(1, 5),
        'b': np.arange(1.0, 2.5, 0.1),
    }
    black_box_fn = run_eval_on_bm25
    all_result = []
    candi = list(generate_hyperparameters(params_range))
    random.shuffle(candi)
    for parameter in candi:
        score = black_box_fn(parameter)
        all_result.append((parameter, score))
        print("{}\t{}".format(parameter, score))

    print("---sorted---")
    all_result.sort(key=get_second)
    for param, score in all_result:
        print("{}\t{}".format(param, score))




if __name__ == "__main__":
    main()