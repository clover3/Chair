from typing import List, Dict

from contradiction.medical_claims.token_tagging.acc_eval.path_helper import load_sbl_binary_label
from contradiction.medical_claims.token_tagging.path_helper import get_save_path2, get_binary_save_path_w_opt
from contradiction.token_tagging.acc_eval.defs import SentTokenLabel, SentTokenBPrediction
from contradiction.token_tagging.acc_eval.eval_codes import calc_prec_rec_acc
from contradiction.token_tagging.acc_eval.parser import save_sent_token_binary_predictions
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry


def convert_to_binary(rlg: Dict[str, List[TrecRankedListEntry]],
                      threshold) -> List[SentTokenBPrediction]:
    output = []
    for qid, entries in rlg.items():
        score_d = {}
        for e in entries:
            score_d[int(e.doc_id)] = 1 if e.score >= threshold else 0

        maybe_len = max(score_d) + 1
        scores = [score_d[i] for i in range(maybe_len)]

        output.append(SentTokenBPrediction(qid, scores))
    return output


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


def build_save(run_name, tag_type, metric_to_opt):
    rl_path = get_save_path2(run_name, tag_type)
    rlg = load_ranked_list_grouped(rl_path)
    labels: List[SentTokenLabel] = load_sbl_binary_label(tag_type, "val")

    def apply_threshold_eval(t):
        predictions: List[SentTokenBPrediction] = convert_to_binary(rlg, t)
        return calc_prec_rec_acc(labels, predictions)

    max_t = None
    max_f1 = -1
    for i in range(102):
        t = 0.01 * i
        metrics = apply_threshold_eval(t)
        if metrics[metric_to_opt] > max_f1:
            max_f1 = metrics[metric_to_opt]
            max_t = t
        # print(t, metrics[metric_to_opt], metrics['precision'], metrics['recall'])
    print("{}={} at t={}".format(metric_to_opt, max_f1, max_t))
    predictions = convert_to_binary(rlg, max_t)
    save_path = get_binary_save_path_w_opt(run_name, tag_type, metric_to_opt)
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
    run_list = ["random", "nlits87", "psearch", "coattention", "word2vec_em",
                "lime",
                "deletion", "exact_match", "word_seg"]
    tag = "conflict"
    # tag = "mismatch"
    metric_to_opt = 'accuracy'

    for run_name in run_list:
        print(run_name)
        try:
            build_save(run_name, tag, metric_to_opt)
        except FileNotFoundError as e:
            print(e)


if __name__ == "__main__":
    main()
