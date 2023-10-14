import os
import sys
import numpy as np
from cache import load_pickle_from
from evals.basic_func import get_acc_prec_recall_i
from typing import List, Iterable, Callable, Dict, Tuple, Set

from tab_print import print_table
from table_lib import tsv_iter
from taskman_client.task_proxy import get_task_manager_proxy


def run_eval_w_tsv_display(prediction_path, tsv_path, target_prediction, target_metric):
    output = load_pickle_from(prediction_path)
    table = list(tsv_iter(tsv_path))
    gold = [int(s) for _, _, s in table]
    majority_pred = np.zeros_like(gold)
    columns = ['accuracy', 'precision', 'recall', 'f1']
    table = []
    eval_targets = [('majority', majority_pred)]
    for k, pred_score in output['align_probe'].items():
        pred_label = np.less(0, pred_score).astype(int)
        eval_targets.append((k, pred_label))
    all_scores = {}
    for method, pred in eval_targets:
        assert len(pred) == len(gold)
        eval_out = get_acc_prec_recall_i(pred, gold)
        row = [method]
        for k in columns:
            all_scores[(method, k)] = eval_out[k]
            row.append(eval_out[k])
        table.append(row)
    print_table(table)
    sig_score = all_scores[target_prediction, target_metric]
    run_name = os.path.basename(prediction_path)


def run_eval_w_tsv_report(prediction_path, tsv_path, target_prediction, target_metric):
    output = load_pickle_from(prediction_path)
    run_name = os.path.basename(prediction_path)
    table = list(tsv_iter(tsv_path))
    gold = [int(s) for _, _, s in table]
    majority_pred = np.zeros_like(gold)
    columns = ['accuracy', 'precision', 'recall', 'f1']
    pred_score = output['align_probe'][target_prediction]
    cut = 0

    pred_label = np.less(cut, pred_score).astype(int)
    eval_out = get_acc_prec_recall_i(pred_label, gold)
    score = eval_out[target_metric]

    proxy = get_task_manager_proxy()
    proxy.report_number(run_name, score, target_prediction, target_metric)


def main():
    prediction_path = sys.argv[1]
    tsv_path = sys.argv[2]
    target_prediction = "align_pred"
    target_metric = "f1"

    run_eval_w_tsv_display(prediction_path, tsv_path, target_prediction, target_metric)


if __name__ == "__main__":
    main()