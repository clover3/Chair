import sys
import numpy as np
from cache import load_pickle_from
from evals.basic_func import get_acc_prec_recall_i
from typing import List, Iterable, Callable, Dict, Tuple, Set

from tab_print import print_table


def main():
    prediction_path = sys.argv[1]
    output = load_pickle_from(prediction_path)

    label = output['label']

    def accuracy(pred):
        n_correct = np.sum(np.equal(pred, label).astype(int))
        n = len(label)
        return n_correct / n

    majority_pred = np.zeros_like(label)

    gold: List[int] = np.reshape(label, [-1]).tolist()
    columns = ['accuracy', 'precision', 'recall', 'f1']
    table = []

    eval_targets = [('majority', majority_pred)]
    for k, v in output['align_probe'].items():
        pred_score = v[:, 1]
        pred_label = np.less(0, pred_score).astype(int)
        eval_targets.append((k, pred_label))

    for method, pred in eval_targets:
        eval_out = get_acc_prec_recall_i(pred, gold)
        row = [method]
        for k in columns:
            row.append(eval_out[k])
        table.append(row)

    print_table(table)


if __name__ == "__main__":
    main()