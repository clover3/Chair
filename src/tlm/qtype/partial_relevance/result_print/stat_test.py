from typing import List

from scipy import stats

from misc_lib import average
from tab_print import tab_print
from tlm.qtype.partial_relevance.eval_metric.meta_common import get_better_fn


def pred_compare(p1: List[int], p2: List[int]):
    return all([a == b for a, b in zip(p1, p2)])


def equal_rate(scores1, scores2):
    is_equal = [0 if s1 > s2 or s1 < s2 else 1 for s1, s2 in zip(scores1, scores2)]
    return is_equal



def do_test_correct_rate(dataset, target_method, metric1, metric2, method_pair,
            get_score_for_method, get_prediction_for_method):
    pred_list_1: List[List[int]] = get_prediction_for_method(dataset, method_pair[0])
    pred_list_2: List[List[int]] = get_prediction_for_method(dataset, method_pair[1])

    def filter_score(scores):
        for i in range(len(scores)):
            s1 = scores[i]
            p1 = pred_list_1[i]
            p2 = pred_list_2[i]
            pred_same = pred_compare(p1, p2)
            if not pred_same:
                yield s1

    def get_correct_list_for(metric):
        better_fn = get_better_fn(metric)
        scores: List[float] = get_score_for_method(dataset, target_method, metric)
        def get_correct(s):
            if better_fn(0.5, s):
                return 1
            else:
                return 0
        return [get_correct(s) for s in filter_score(scores)]

    correct_list_1 = get_correct_list_for(metric1)
    correct_list_2 = get_correct_list_for(metric2)

    diff, p = stats.ttest_rel(correct_list_1, correct_list_2)
    tab_print(metric1, metric2, target_method,
              average(correct_list_1), average(correct_list_2), len(correct_list_1), diff, p)


def do_test_preference(dataset, target_decision, metric1, metric2, method_pair,
                         get_score_for_method, get_prediction_for_method):
    pred_list_1: List[List[int]] = get_prediction_for_method(dataset, method_pair[0])
    pred_list_2: List[List[int]] = get_prediction_for_method(dataset, method_pair[1])

    def filter_score(scores):
        for i in range(len(scores)):
            s1 = scores[i]
            p1 = pred_list_1[i]
            p2 = pred_list_2[i]
            pred_same = pred_compare(p1, p2)
            if not pred_same:
                yield s1

    def get_preference_on(metric):
        def get_score_inner(method):
            scores: List[float] = get_score_for_method(dataset, method, metric)
            scores = list(filter_score(scores))
            return scores
        scores1: List[float] = get_score_inner(method_pair[0])
        scores2: List[float] = get_score_inner(method_pair[1])
        better_fn = get_better_fn(metric)
        def get_preference(s_pair):
            s1, s2 = s_pair
            s2_better = better_fn(s1, s2)
            s1_better = better_fn(s2, s1)
            if s1_better:
                decision = "left"
            elif s2_better:
                decision = "right"
            else:
                decision = "equal"
            return 1 if decision == target_decision else 0
        return [get_preference(s) for s in zip(scores1, scores2)]

    preference_rate1 = get_preference_on(metric1)
    preference_rate2 = get_preference_on(metric2)

    diff, p = stats.ttest_rel(preference_rate1, preference_rate2)
    if p < 0.01:
        print(metric1, metric2, target_decision)
    # tab_print(metric1, metric2, target_decision,
    #           average(preference_rate1), average(preference_rate2), len(preference_rate2), diff, p)



