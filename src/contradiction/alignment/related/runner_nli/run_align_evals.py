from contradiction.alignment.related.align_eval import run_eval
from misc_lib import tprint


def main():
    dataset = "dev"
    method_list = ["exact_match", "random"]
    metric_list = [
                  "replace_suff_v3",
                    "replace_suff_v3d",
                   "erasure_v3", "erasure_v3d",
                   "erasure_suff_v3", "erasure_suff_v3d",
                   "replace_v3", "replace_v3d",
                   # "attn_v3"
                   ]
    for metric in metric_list:
        for method in method_list:
            tprint(metric, method)
            run_eval(dataset, method, metric, "localhost")


if __name__ == "__main__":
    main()