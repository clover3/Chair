
import os.path

from contradiction.mnli_ex.path_helper import get_save_path, get_save_path_ex, get_mnli_ex_trec_style_label_path, \
    get_binary_save_path_w_opt
from contradiction.token_tagging.acc_eval.defs import SentTokenLabel
from contradiction.token_tagging.acc_eval.eval_codes import calc_prec_rec_acc
from contradiction.token_tagging.acc_eval.parser import load_sent_token_binary_predictions
from runnable.trec.trec_eval_like import trec_eval_like_func
from tab_print import print_table
from trec.trec_eval_wrap_fn import run_trec_eval_parse
from typing import List

from contradiction.mnli_ex.load_mnli_ex_data import load_mnli_ex_binary_label


def compute_binary_metrics_for_test(run_name, tag_type, metric):
    labels: List[SentTokenLabel] = load_mnli_ex_binary_label("test", tag_type)
    save_path = get_binary_save_path_w_opt(run_name, tag_type, metric)
    if os.path.exists(save_path):
        predictions = load_sent_token_binary_predictions(save_path)
        metrics = calc_prec_rec_acc(labels, predictions)
    else:
        metrics = {metric: '-'}
    return metrics


class MetricCalc:
    def __init__(self, split):
        self.split = split
        self.expected_num_q_d = {
            ("val", "mismatch"): NotImplemented,
            ("val", "mismatch"): NotImplemented,
            ("test", "mismatch"): NotImplemented,
            ("test", "conflict"): NotImplemented,
        }

    def compute(self, run_name, tag, metric):
        prediction_path = get_save_path_ex(self.split, run_name, tag)
        qrel_path = get_mnli_ex_trec_style_label_path(tag, self.split)
        try:
            if metric == "MAP":
                if not os.path.exists(prediction_path):
                    raise FileNotFoundError(prediction_path)
                score_d = run_trec_eval_parse(prediction_path, qrel_path)
                s = score_d["map"]
                n_q = score_d["num_q"]
                print(f"{n_q} queries")
            elif metric == "P1":
                if not os.path.exists(prediction_path):
                    raise FileNotFoundError(prediction_path)
                s = trec_eval_like_func(qrel_path, prediction_path, metric)
            elif metric == "accuracy":
                score_d = compute_binary_metrics_for_test(run_name, tag, metric)
                s = score_d[metric]
            else:
                raise Exception()
        except FileNotFoundError as e:
            print(e)
            s = "-"
        return s


def show_all():
    run_list = ["token_entail",
                # "word2vec_em",
                "nlits87",
                ]
    split = "test"

    scorer = MetricCalc(split)
    head = ["run name",
            "conflict", "", "",
            "match", "", "",
            "mismatch", "", "",
            ]
    table = [head]
    metric_list = ["P1", "MAP", "accuracy",]
    for run_name in run_list:
        row = [run_name]
        for tag in ["conflict", "match", "mismatch"]:
            for metric in metric_list:
                s = scorer.compute(run_name, tag, metric)
                row.append(s)
        table.append(row)
    print_table(table)


if __name__ == "__main__":
    show_all()
