import sys
import numpy as np
from evals.basic_func import get_acc_prec_recall_i
from table_lib import tsv_iter
from taskman_client.task_proxy import get_task_manager_proxy


def run_eval_w_tsv(lines_scores_path, tsv_path, target_metric, cut, run_name=None):
    pred_score = [float(s) for s in open(lines_scores_path, "r")]
    table = list(tsv_iter(tsv_path))
    gold = [int(s) for _, _, s in table]
    assert len(pred_score) == len(gold)

    pred_label = np.less(cut, pred_score).astype(int)
    eval_out = get_acc_prec_recall_i(pred_label, gold)
    score = eval_out[target_metric]
    print(score)

    if run_name:
        proxy = get_task_manager_proxy()
        proxy.report_number(run_name, score, "", target_metric)


def main():
    tsv_path = sys.argv[1]
    lines_scores_path = sys.argv[2]
    cut = float(sys.argv[3])
    try:
        run_name = sys.argv[4]
    except IndexError:
        run_name = None
    target_metric = "f1"
    run_eval_w_tsv(lines_scores_path, tsv_path, target_metric, cut, run_name)


if __name__ == "__main__":
    main()