

import os.path

from contradiction.esnli.path_helper import get_save_path_ex, get_binary_save_path_w_opt, load_esnli_binary_label, \
    load_esnli_binary_label_all
from contradiction.token_tagging.acc_eval.defs import SentTokenLabel
from contradiction.token_tagging.acc_eval.eval_codes import calc_prec_rec_acc
from contradiction.token_tagging.acc_eval.parser import load_sent_token_binary_predictions
from tab_print import print_table
from typing import List


def compute_binary_metrics_for_test(run_name, target_sent, metric):
    labels: List[SentTokenLabel] = load_esnli_binary_label_all("test",)
    save_path = get_binary_save_path_w_opt(run_name, target_sent, metric)
    if os.path.exists(save_path):
        predictions = load_sent_token_binary_predictions(save_path)
        metrics = calc_prec_rec_acc(labels, predictions)
    else:
        metrics = {metric: '-'}
    return metrics


def show_all():
    run_list = ["token_entail",
                "nlits87",
                # "lime",
                ]
    split = "test"
    head = ["run name",
            "prem", "", "",
            "hypo", "", "",
            ]
    table = [head]
    metric_list = ["precision", "recall", "f1"]
    for run_name in run_list:
        row = [run_name]
        for target_sent in ["prem", "hypo"]:
            score_d = compute_binary_metrics_for_test(run_name, target_sent, "f1")
            for metric in metric_list:
                s = score_d[metric]
                row.append(s)
        table.append(row)
    print_table(table)


if __name__ == "__main__":
    show_all()
