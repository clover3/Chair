import os

from cache import load_list_from_jsonl
from contradiction.medical_claims.token_tagging.acc_eval.path_helper import load_sbl_binary_label
from contradiction.token_tagging.acc_eval.defs import SentTokenLabel, SentTokenBPrediction
from contradiction.token_tagging.acc_eval.eval_codes import calc_prec_rec_acc_flat
from cpath import output_path
from list_lib import lflatten
from misc_lib import exist_or_mkdir
from tab_print import print_table
from typing import List, Iterable, Callable, Dict, Tuple, Set


def get_binary_save_path_w_opt(run_name, tag_type, metric):
    tag_type = {
        "mismatch": "neutral",
        "conflict": "contradiction"
    }[tag_type]
    save_name = "test_{}_{}".format(run_name, tag_type)
    dir_save_path = os.path.join(output_path, "alamri_annotation1", "binary_predictions_new_code")
    exist_or_mkdir(dir_save_path)
    save_path = os.path.join(dir_save_path, save_name + ".txt")
    return save_path


def load_sent_token_binary_predictions(tag, save_path):
    def from_json(j):
        pair_id = j['pair_id']
        sent1_pred = j['sent1_pred']
        sent2_pred = j['sent2_pred']
        yield SentTokenBPrediction(f"{pair_id}_prem_{tag}", sent1_pred)
        yield SentTokenBPrediction(f"{pair_id}_hypo_{tag}", sent2_pred)

    return lflatten(load_list_from_jsonl(save_path, from_json))


def compute_binary_metrics_for_test(run_name, tag_type, metric):
    labels: List[SentTokenLabel] = load_sbl_binary_label(tag_type, "test")
    save_path = get_binary_save_path_w_opt(run_name, tag_type, metric)
    print(save_path)
    if os.path.exists(save_path):
        predictions = load_sent_token_binary_predictions(tag_type, save_path)
        metrics = calc_prec_rec_acc_flat(labels, predictions)
    else:
        print("Not found")
        metrics = {metric: '-'}
    return metrics


def main():
    run_name = "nlits87"
    metric = "f1"
    # tag = "mismatch"
    tag = "conflict"
    print(metric)
    def number_display(n):
        if type(n) == float:
            return "{0:.3f}".format(n)
        else:
            return n

    table = []
    all_metric_names = ["accuracy", "f1", "precision", "recall"]
    table.append(["run name"] + all_metric_names)
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


if __name__ == "__main__":
    main()
