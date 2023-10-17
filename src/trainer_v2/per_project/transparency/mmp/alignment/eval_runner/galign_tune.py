import sys
import numpy as np
from cache import load_pickle_from
from evals.basic_func import get_acc_prec_recall_i
from table_lib import tsv_iter


def run_tune(lines_scores_path, tsv_path, target_metric):
    pred_score = [float(s) for s in open(lines_scores_path, "r")]

    table = list(tsv_iter(tsv_path))
    gold = [int(s) for _, _, s in table]
    assert len(pred_score) == len(gold)

    st = -2
    ed = 6
    step = 0.2

    cut = st
    max_cut = cut
    max_score = -100
    print("Cut\tF1\tPre\tRecall")
    while cut <= ed:
        pred_label = np.less(cut, pred_score).astype(int)
        eval_out = get_acc_prec_recall_i(pred_label, gold)
        score = eval_out[target_metric]
        print(f"{cut}\t{score:.4f}\t{eval_out['precision']}\t{eval_out['recall']}")
        if score > max_score:
            max_score = score
            max_cut = cut
        cut += step
    return max_cut

