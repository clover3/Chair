from typing import List, Iterable, Callable, Dict, Tuple, Set

from contradiction.medical_claims.token_tagging.acc_eval.path_helper import load_sbl_binary_label
from contradiction.medical_claims.token_tagging.path_helper import get_flat_binary_save_path, mismatch_only_method_list, \
    get_binary_save_path_w_opt, get_debug_qid_save_path
from contradiction.medical_claims.token_tagging.print_score.stat_test import load_bin_prediction_tuned_f1
from contradiction.token_tagging.acc_eval.defs import SentTokenBPrediction, SentTokenLabel
from contradiction.token_tagging.acc_eval.parser import load_sent_token_binary_predictions


def save_line_scores(scores, save_path):
    with open(save_path, "w") as f:
        for s in scores:
            f.write("{}\n".format(s))


def main():
    run_list = ["exact_match", "word2vec_em","coattention",
                "lime", "deletion","senli", "word_seg",
                "slr",
                "nlits87",
                 "davinci", 'gpt-3.5-turbo'
                 ]

    for tag in ["mismatch", "conflict"]:
        labels: List[SentTokenLabel] = load_sbl_binary_label(tag, "test")
        qid_test = [p.qid for p in labels]
        for metric in ["f1", "accuracy"]:
            for run_name in run_list:
                qid_ordering = []
                try:
                    save_path = get_binary_save_path_w_opt(run_name, tag, metric)
                    pred: List[SentTokenBPrediction] = load_sent_token_binary_predictions(save_path)
                    pred.sort(key=lambda x: x.qid)
                    pred_flat_list = []
                    for per_sent in pred:
                        if per_sent.qid in qid_test:
                            qid_ordering.append(per_sent.qid)
                            pred_flat_list.extend(per_sent.predictions)

                    save_path = get_flat_binary_save_path(run_name, tag, metric)
                    save_line_scores(pred_flat_list, save_path)

                    f = open(get_debug_qid_save_path(run_name, tag, metric), "w")
                    for qid in qid_ordering:
                        f.write(f"{qid}\n")

                except FileNotFoundError as e:
                    if run_name not in mismatch_only_method_list:
                        raise e


if __name__ == "__main__":
    main()