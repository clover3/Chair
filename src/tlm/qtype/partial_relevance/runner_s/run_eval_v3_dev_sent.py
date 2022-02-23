from tlm.qtype.partial_relevance.runner.run_eval.run_eval_v3 import run_eval


def main():
    model_interface = "localhost"
    dataset = "dev_sent"
    dataset = "dev_sw"
    method_list = ["exact_match", "random_cut"] #, "exact_match_noise0.1"]
    # metric_list = ["erasure_v3", "attn_v3", "replace_v3"]
    metric_list = ["replace_suff_v3", "replace_suff_v3d",
                   "erasure_v3", "erasure_v3d",
                   "erasure_suff_v3", "erasure_suff_v3d",
                   "replace_v3", "replace_v3d",
                   # "attn_v3"
                ]
    for metric in metric_list:
        for method in method_list:
            # if is_run_exist(dataset, method, metric):
            #     print("Skip ", dataset, method, metric)
            # else:
            run_eval(dataset, method, metric, model_interface)


if __name__ == "__main__":
    main()
