
from typing import List

from contradiction.esnli.path_helper import get_save_path_ex
from contradiction.token_tagging.acc_eval.defs import SentTokenLabel, convert_to_binary, SentTokenBPrediction
from contradiction.token_tagging.acc_eval.eval_codes import calc_prec_rec_acc
from contradiction.token_tagging.acc_eval.parser import save_sent_token_binary_predictions
from data_generator.NLI.enlidef import enli_tags
from trec.trec_parse import load_ranked_list_grouped



def build_save(run_name, tag_type, metric_to_opt):
    dev_rl_path = get_save_path_ex("dev", run_name, tag_type)
    dev_rlg = load_ranked_list_grouped(dev_rl_path)
    labels: List[SentTokenLabel] = load_esnli_binary_label("dev", tag_type)

    def apply_threshold_eval(t):
        predictions: List[SentTokenBPrediction] = convert_to_binary(dev_rlg, t)
        return calc_prec_rec_acc(labels, predictions)

    max_t = None
    max_score = -1
    for i in range(102):
        t = 0.01 * i
        metrics = apply_threshold_eval(t)
        if metrics[metric_to_opt] > max_score:
            max_score = metrics[metric_to_opt]
            max_t = t
        # print(t, metrics[metric_to_opt], metrics['precision'], metrics['recall'])
    print("{}={} at t={}".format(metric_to_opt, max_score, max_t))

    test_rl_path = get_save_path_ex("test", run_name, tag_type)
    test_rlg = load_ranked_list_grouped(test_rl_path)

    predictions = convert_to_binary(test_rlg, max_t)
    save_path = get_binary_save_path_w_opt(run_name, tag_type, metric_to_opt)
    save_sent_token_binary_predictions(predictions, save_path)


def main():
    run_list = ["token_entail", ]
    run_list = ["nlits87", ]
    metric_to_opt = 'accuracy'
    for run_name in run_list:
        for tag in enli_tags:
            print(run_name)
            try:
                build_save(run_name, tag, metric_to_opt)
            except FileNotFoundError as e:
                print(e)


if __name__ == "__main__":
    main()
