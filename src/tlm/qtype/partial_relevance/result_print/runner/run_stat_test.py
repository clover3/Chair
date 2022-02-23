from tlm.qtype.partial_relevance.result_print.runner.show_preference_v3 import get_prediction_for_method, \
    get_score_for_method
from tlm.qtype.partial_relevance.result_print.stat_test import do_test_correct_rate, do_test_preference


def main2():
    method_list = ["exact_match", "random_cut"]
    metric_pair_list = [
                   ("erasure_v3d", "replace_v3d"),
                   ("erasure_suff_v3d", "replace_suff_v3d"),
                   ("erasure_v3", "replace_v3"),
                   ("erasure_suff_v3", "replace_suff_v3"),
                   ]
    # get_prediction_for_method, get_score_for_method = all_func
    dataset = "dev_sw"
    iterate_do_test_correct_rate(dataset, method_list, metric_pair_list)


def iterate_do_test_correct_rate(dataset, method_list, metric_pair_list):
    for method in method_list:
        for pair in metric_pair_list:
            metric1, metric2 = pair
            do_test_correct_rate(dataset, method, metric1, metric2, method_list,
                                    get_score_for_method,
                                    get_prediction_for_method)


def main3():
    method_list = ["exact_match", "random_cut"]
    metric_pair_list = [
        ("erasure_v3d", "replace_v3d"),
        ("erasure_suff_v3d", "replace_suff_v3d"),
        ("erasure_v3", "replace_v3"),
        ("erasure_suff_v3", "replace_suff_v3"),
    ]
    # get_prediction_for_method, get_score_for_method = all_func
    dataset = "dev_sw"
    iterate_do_test_correct_rate(dataset, method_list, metric_pair_list)


def main():
    method_list = ["exact_match", "random_cut"]
    metric_pair_list = [
        ("erasure_v3d", "replace_v3d"),
        ("erasure_suff_v3d", "replace_suff_v3d"),
        ("erasure_v3", "replace_v3"),
        ("erasure_suff_v3", "replace_suff_v3"),
    ]
    dataset = "dev_sw"
    for decision in ["left", "right", "equal"]:
        for pair in metric_pair_list:
            metric1, metric2 = pair
            do_test_preference(dataset, decision, metric1, metric2, method_list,
                               get_score_for_method,
                               get_prediction_for_method)


if __name__ == "__main__":
    main()
