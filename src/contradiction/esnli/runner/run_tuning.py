
from typing import List

from contradiction.esnli.path_helper import get_save_path_ex, load_esnli_binary_label, get_binary_save_path_w_opt, \
    load_esnli_binary_label_all
from contradiction.token_tagging.acc_eval.defs import SentTokenLabel, convert_to_binary, SentTokenBPrediction
from contradiction.token_tagging.acc_eval.eval_codes import calc_prec_rec_acc
from contradiction.token_tagging.acc_eval.parser import save_sent_token_binary_predictions
from data_generator.NLI.enlidef import enli_tags
from trec.trec_parse import load_ranked_list_grouped


def load_run(split, run_name, target_sent=""):
    rlg_all = {}
    for tag in enli_tags:
        dev_rl_path = get_save_path_ex(split, run_name, tag)
        dev_rlg = load_ranked_list_grouped(dev_rl_path)
        rlg_all.update(dev_rlg)

    if target_sent:
        new_rlg = {}
        for qid in rlg_all:
            if qid.endswith(target_sent):
                new_rlg[qid] = rlg_all[qid]
        rlg_all = new_rlg

    return rlg_all


def build_save(run_name, metric_to_opt, target_sent):
    dev_rlg_all = load_run("dev", run_name, target_sent)

    print("{} queries".format(len(dev_rlg_all)))
    labels: List[SentTokenLabel] = load_esnli_binary_label_all("dev")
    print("{} Labels ".format(len(labels)))

    def apply_threshold_eval(t):
        predictions: List[SentTokenBPrediction] = convert_to_binary(dev_rlg_all, t)
        return calc_prec_rec_acc(labels, predictions)

    max_t = None
    max_score = -1
    for i in range(-100, 102, 2):
        t = 0.01 * i
        metrics = apply_threshold_eval(t)
        if metrics[metric_to_opt] > max_score:
            max_score = metrics[metric_to_opt]
            max_t = t
    print("{}={} at t={}".format(metric_to_opt, max_score, max_t))
    test_rlg = load_run("test", run_name, target_sent)
    predictions = convert_to_binary(test_rlg, max_t)
    save_path = get_binary_save_path_w_opt(run_name, target_sent, metric_to_opt)
    save_sent_token_binary_predictions(predictions, save_path)



def main():
    run_list = ["nlits87","token_entail", ]
    metric_to_opt = 'f1'
    for run_name in run_list:
        for target_sent in ["prem", "hypo"]:
            print(run_name)
            try:
                build_save(run_name, metric_to_opt, target_sent)
            except FileNotFoundError as e:
                print(e)


if __name__ == "__main__":
    main()
