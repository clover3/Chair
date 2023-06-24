import os.path
from typing import List

from contradiction.medical_claims.token_tagging.acc_eval.path_helper import load_sbl_binary_label
from contradiction.medical_claims.token_tagging.path_helper import get_binary_save_path_w_opt, get_binary_save_path
from contradiction.token_tagging.acc_eval.defs import SentTokenLabel
from contradiction.token_tagging.acc_eval.eval_codes import calc_prec_rec_acc, calc_prec_rec_acc_flat
from contradiction.token_tagging.acc_eval.parser import load_sent_token_binary_predictions
from tab_print import print_table


def compute_binary_metrics_for_test(run_name, tag_type, metric):
    labels: List[SentTokenLabel] = load_sbl_binary_label(tag_type, "test")
    if run_name in ["davinci", "gpt-3.5-turbo", "exact_match"]:
        save_path = get_binary_save_path(run_name, tag_type)
    else:
        save_path = get_binary_save_path_w_opt(run_name, tag_type, metric)
    if os.path.exists(save_path):
        predictions = load_sent_token_binary_predictions(save_path)
        metrics = calc_prec_rec_acc_flat(labels, predictions)
    else:
        print(save_path)
        metrics = {metric: '-'}
    return metrics


def print_scores(metric, run_list, tag):
    print(metric)
    def number_display(n):
        if type(n) == float:
            return "{0:.3f}".format(n)
        else:
            return n

    table = []
    all_metric_names = ["accuracy", "precision", "recall", "f1", "tp", "fp", "tn", "fn"]
    all_metric_names = ["accuracy", "f1", "precision", "recall"]
    table.append(["run name"] + all_metric_names)
    for run_name in run_list:
        try:
            scores_d = compute_binary_metrics_for_test(run_name, tag, metric)
            scores = []
            for m in all_metric_names:
                try:
                    scores.append(scores_d[m])
                except KeyError as e:
                    scores.append('-')

            row = [run_name]
            row.extend(map(number_display, scores))

            table.append(row)
        except FileNotFoundError as e:
            print(e)
    print_table(table)


def main():
    run_list = ["random", "exact_match",
                "coattention", "word_seg",
                "nlits87",
                "davinci",
                "gpt-3.5-turbo"
                ]

    metric = "f1"
    # tag = "mismatch"
    tag = "conflict"
    print_scores(metric, run_list, tag)


if __name__ == "__main__":
    main()
