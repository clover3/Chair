import itertools
from typing import List

from scipy.stats import ttest_rel

from misc_lib import average
from tlm.qtype.partial_relevance.calc_avg import load_eval_result_r, load_eval_result_b


def get_nonnull_scores(eval_res):
    scores: List[float] = [t[1] for t in eval_res if t[1] is not None]
    return scores


def main():
    dataset_list = ["dev", "devp", "devn"]
    method_list = ["random", "gradient", "attn_perturbation"]
    metric_list = ["partial_relevant", "erasure"]
    print_paired_ttest(dataset_list, method_list, metric_list)


def count_high_low_win(scores1, scores2):
    high_win = 0
    low_win = 0
    for s1, s2 in zip(scores1, scores2):
        if s1 > s2:
            high_win += 1
        elif s2 > s1:
            low_win += 1

    return high_win, low_win


def print_paired_binary(dataset_list, method_list, metric_list):
    for dataset in dataset_list:
        print(dataset)
        for metric in metric_list:
            def get_score_for_method(method):
                run_name = "{}_{}_{}".format(dataset, method, metric)
                eval_res = load_eval_result_r(run_name)
                scores = get_nonnull_scores(eval_res)
                return scores

            n_method = len(method_list)
            pairs = list(itertools.combinations(list(range(n_method)), 2))
            for i1, i2 in pairs:
                method1 = method_list[i1]
                method2 = method_list[i2]
                scores1 = get_score_for_method(method1)
                scores2 = get_score_for_method(method2)
                left_better = average(scores1) - average(scores2) > 0
                if left_better:
                    high_win, low_win = count_high_low_win(scores1, scores2)
                else:
                    high_win, low_win = count_high_low_win(scores2, scores1)
                if left_better:
                    method1_s = method1
                    method2_s = method2
                else:
                    method1_s = method2
                    method2_s = method1
                print("{0}\t{1} > {2}\t{3}\t{4}".format(metric, method1_s,
                                                       method2_s,
                                                       # two_digit_float(avg_gap),
                                                       high_win, low_win))


def print_paired_ttest_inner(dataset_list, method_list, metric_list, load_eval_result_fn):
    for dataset in dataset_list:
        print(dataset)
        for metric in metric_list:
            def get_score_for_method(method):
                run_name = "{}_{}_{}".format(dataset, method, metric)
                eval_res = load_eval_result_fn(run_name)
                scores = get_nonnull_scores(eval_res)
                return scores

            n_method = len(method_list)
            pairs = list(itertools.combinations(list(range(n_method)), 2))
            for i1, i2 in pairs:
                method1 = method_list[i1]
                method2 = method_list[i2]
                scores1 = get_score_for_method(method1)
                scores2 = get_score_for_method(method2)
                left_better = average(scores1) - average(scores2) > 0
                avg_gap, p_value = ttest_rel(scores1, scores2)
                if left_better:
                    method1_s = method1
                    method2_s = method2
                else:
                    method1_s = method2
                    method2_s = method1
                print("{0}\t{1} > {2}\t{3:.4f}".format(metric, method1_s,
                                           method2_s,
                                           # two_digit_float(avg_gap),
                                           p_value))


def print_paired_ttest_r(dataset_list, method_list, metric_list):
    return print_paired_ttest_inner(dataset_list, method_list, metric_list, load_eval_result_r)


def print_paired_ttest_b(dataset_list, method_list, metric_list):
    return print_paired_ttest_inner(dataset_list, method_list, metric_list, load_eval_result_b)


def main2():
    dataset_list = ["dev_word", "dev_wordp", "dev_wordn"]
    # method_list = ["random", "gradient", "attn_perturbation"]
    method_list = ["random", "exact_match", "gradient",]
    metric_list = ["partial_relevant", "erasure"]
    print_paired_ttest_r(dataset_list, method_list, metric_list)
    # print_paired_binary(dataset_list, method_list, metric_list)


def main3():
    dataset_list = ["dev_sent"]
    method_list = ["exact_match", "exact_match_noise0.1", ]
    for option in ["precision", "recall"]:
        for target_idx in [0, 1]:
            metric_list = []
            for wordset in ["emptyword", "100words"]:
                metric = f"replace_{option}_{wordset}_{target_idx}"
                metric_list.append(metric)
            print_paired_ttest_r(dataset_list, method_list, metric_list)
    # print_paired_binary(dataset_list, method_list, metric_list)



if __name__ == "__main__":
    main3()
