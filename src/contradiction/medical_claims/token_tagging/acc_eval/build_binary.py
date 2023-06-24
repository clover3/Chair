from typing import List

from contradiction.medical_claims.token_tagging.acc_eval.path_helper import load_sbl_binary_label
from contradiction.medical_claims.token_tagging.path_helper import get_save_path2, get_binary_save_path_w_opt, \
    get_binary_save_path
from contradiction.token_tagging.acc_eval.defs import SentTokenLabel, SentTokenBPrediction, convert_to_binary
from contradiction.token_tagging.acc_eval.eval_codes import calc_prec_rec_acc
from contradiction.token_tagging.acc_eval.parser import save_sent_token_binary_predictions
from trec.trec_parse import load_ranked_list_grouped


def build_save(run_name, tag_type, threshold):
    rl_path = get_save_path2(run_name, tag_type)
    rlg = load_ranked_list_grouped(rl_path)
    predictions = convert_to_binary(rlg, threshold)
    save_path = get_binary_save_path(run_name, tag_type)
    save_sent_token_binary_predictions(predictions, save_path)


def main():
    # run_list = ["exact_match", "gpt-3.5-turbo"]
    run_list = ["exact_match", "gpt-3.5-turbo", "davinci"]
    # run_list = ["slr"]
    tag_list = ["mismatch", "conflict"]
    for run_name in run_list:
        for tag in tag_list:
            print(run_name, tag)
            try:
                build_save(run_name, tag, 0.5)
            except FileNotFoundError as e:
                print(e)


if __name__ == "__main__":
    main()
