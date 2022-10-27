from typing import List

from contradiction.medical_claims.token_tagging.acc_eval.path_helper import load_sbl_binary_label
from contradiction.medical_claims.token_tagging.path_helper import get_binary_save_path, get_binary_save_path_w_opt
from contradiction.token_tagging.acc_eval.label_loaders import SentTokenLabel, load_sent_token_binary_predictions, \
    calc_prec_rec_acc


def show_test_score(run_name, tag_type, metric):
    labels: List[SentTokenLabel] = load_sbl_binary_label(tag_type, "test")
    save_path = get_binary_save_path_w_opt(run_name, tag_type, metric)
    predictions = load_sent_token_binary_predictions(save_path)
    metrics = calc_prec_rec_acc(labels, predictions)
    return metrics


def show_for_conflict():
    run_list = ["random", "nlits86", "psearch", "senli", "deletion", "exact_match"]
    metric = "f1"
    tag = "conflict"
    print_scores(metric, run_list, tag)


def print_scores(metric, run_list, tag):
    print(metric)
    for run_name in run_list:
        scores = show_test_score(run_name, tag, metric)
        print("{}\t{}".format(run_name, scores[metric]))


def show_for_mismatch():
    mismatch_run_list = ["random", "nlits86", "tf_idf", "psearch", "coattention", "senli", "deletion", "exact_match"]
    metric = "f1"
    tag = "mismatch"
    print_scores(metric, mismatch_run_list, tag)


if __name__ == "__main__":
    show_for_conflict()