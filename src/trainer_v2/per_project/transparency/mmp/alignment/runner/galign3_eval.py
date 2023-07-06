import os
import sys
import numpy as np
from cache import load_pickle_from
from evals.basic_func import get_acc_prec_recall_i
from typing import List, Iterable, Callable, Dict, Tuple, Set

from tab_print import print_table
from taskman_client.task_proxy import get_task_manager_proxy


def main():
    prediction_path = sys.argv[1]
    output = load_pickle_from(prediction_path)

    label = output['label']
    majority_pred = np.zeros_like(label)

    gold: List[int] = np.reshape(label, [-1]).tolist()
    columns = ['accuracy', 'precision', 'recall', 'f1']
    table = []

    eval_targets = [('majority', majority_pred)]
    for k, pred_score in output['align_probe'].items():
        pred_label = np.less(0, pred_score).astype(int)
        eval_targets.append((k, pred_label))

    all_scores = {}
    for method, pred in eval_targets:
        eval_out = get_acc_prec_recall_i(pred, gold)
        row = [method]
        for k in columns:
            all_scores[(method, k)] = eval_out[k]
            row.append(eval_out[k])
        table.append(row)

    print_table(table)

    sig_score = all_scores["g_attention_output", "f1"]
    run_name = os.path.basename(prediction_path)
    proxy = get_task_manager_proxy()
    proxy.report_number(run_name, sig_score, "g_attention_output", "f1")


if __name__ == "__main__":
    main()