import sys
import numpy as np
from cache import load_pickle_from
from evals.basic_func import get_acc_prec_recall_i
from table_lib import tsv_iter


def run_eval_w_tsv(prediction_path, tsv_path, target_prediction, target_metric):
    output = load_pickle_from(prediction_path)
    table = list(tsv_iter(tsv_path))
    gold = [int(s) for _, _, s in table]
    pred_score = output['align_probe'][target_prediction]
    assert len(pred_score) == len(gold)

    st = -2
    ed = 6
    step = 0.2

    cut = st
    print("Cut\tF1\tPre\tRecall")
    while cut <= ed:
        pred_label = np.less(cut, pred_score).astype(int)
        eval_out = get_acc_prec_recall_i(pred_label, gold)
        score = eval_out[target_metric]
        print(f"{cut}\t{score:.4f}\t{eval_out['precision']}\t{eval_out['recall']}")
        cut += step


def main():
    prediction_path = sys.argv[1]
    tsv_path = sys.argv[2]
    target_prediction = "align_pred"
    target_metric = "f1"

    run_eval_w_tsv(prediction_path, tsv_path, target_prediction, target_metric)


if __name__ == "__main__":
    main()