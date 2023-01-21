from typing import List

from contradiction.medical_claims.token_tagging.acc_eval.path_helper import load_sbl_binary_label
from contradiction.medical_claims.token_tagging.path_helper import get_save_path2, get_binary_save_path_w_opt
from contradiction.token_tagging.acc_eval.defs import SentTokenLabel, SentTokenBPrediction, convert_to_binary
from contradiction.token_tagging.acc_eval.eval_codes import calc_prec_rec_acc
from contradiction.token_tagging.acc_eval.parser import save_sent_token_binary_predictions
from trec.trec_parse import load_ranked_list_grouped


def pairing_check(stl_list: List[SentTokenLabel]):
    prem_set = set()
    hypo_set = set()

    for stl in stl_list:
        group_no, problem_no, sent_type, tag = stl.qid.split("_")
        if sent_type == "prem":
            prem_set.add((group_no, problem_no))
        if sent_type == "hypo":
            hypo_set.add((group_no, problem_no))

    for group_no, problem_no in prem_set:
        if (group_no, problem_no) not in hypo_set:
            print(group_no, problem_no, "has no hypo")
    for group_no, problem_no in hypo_set:
        if (group_no, problem_no) not in prem_set:
            print(group_no, problem_no, "has no prem")


def build_save(src_run_name, out_run_name, tag_type, metric_to_opt):
    rl_path = get_save_path2(src_run_name, tag_type)
    rlg = load_ranked_list_grouped(rl_path)

    max_t = +10e8
    predictions = convert_to_binary(rlg, max_t)
    save_path = get_binary_save_path_w_opt(out_run_name, tag_type, metric_to_opt)
    save_sent_token_binary_predictions(predictions, save_path)


def show_true_rate():
    labels: List[SentTokenLabel] = load_sbl_binary_label("mismatch", "val")
    n_true_sum = 0
    n_sum = 0
    for label in labels:
        n_true = sum(label.labels)
        n = len(label.labels)
        n_true_sum += n_true
        n_sum += n

    print("{}/{} = {}".format(n_true_sum, n_sum, n_true_sum / n_sum))


def main():
    # show_true_rate()
    run_list = ["lime"]
    # tag = "conflict"
    tag = "mismatch"
    metric_to_opt = 'f1'
    # metric_to_opt = 'accuracy'
    out_run_name = "majority"
    for run_name in run_list:
        print(run_name)
        try:
            build_save(run_name, out_run_name, tag, metric_to_opt)
        except FileNotFoundError as e:
            print(e)


if __name__ == "__main__":
    main()
