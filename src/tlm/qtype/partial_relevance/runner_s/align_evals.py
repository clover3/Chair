import os

from tlm.qtype.partial_relevance.eval_score_dp_helper import get_eval_score_save_path_b_single
from tlm.qtype.partial_relevance.runner.run_eval.run_align_eval_b import run_align_eval_b


def main():
    dataset = "dev_sent"
    method_list = ["exact_match", "gradient", "random", "exact_match_noise0.1"]
    # policy_name_list = ["attn", "ps_replace_precision", "ps_replace_recall"]
    policy_name_list = ["ps_deletion_precision", "ps_deletion_recall"]
    model_interface = "localhost"

    for method in method_list:
        for policy_name in policy_name_list:
            run_name = "{}_{}_{}".format(dataset, method, policy_name)
            print("run_name", run_name)
            save_path = get_eval_score_save_path_b_single(run_name)
            if not os.path.exists(save_path):
                run_align_eval_b(dataset, method, policy_name, model_interface)


if __name__ == "__main__":
    main()
