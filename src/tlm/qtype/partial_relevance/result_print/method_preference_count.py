import itertools
from collections import Counter
from typing import List

from tab_print import print_table
from tlm.qtype.partial_relevance.eval_metric.meta_common import get_better_fn


def pred_compare(p1: List[int], p2: List[int]):
    return all([a == b for a, b in zip(p1, p2)])


def count_paired_comparison(dataset_list, method_list, metric_list,
                            get_score_for_method, get_prediction_for_method):
    for dataset in dataset_list:
        print(dataset)
        head = ["metric", "method_left", "method_right"]
        decisions = ["left", "right", "equal", "left correct", "right correct"]
        head += decisions
        table = [head]
        for metric in metric_list:
            n_method = len(method_list)
            pairs = list(itertools.combinations(list(range(n_method)), 2))
            for i1, i2 in pairs:
                method1 = method_list[i1]
                method2 = method_list[i2]
                print(method1, method2, metric)
                scores1: List[float] = get_score_for_method(dataset, method1, metric)
                scores2: List[float] = get_score_for_method(dataset, method2, metric)
                pred_list_1: List[List[int]] = get_prediction_for_method(dataset, method1)
                pred_list_2: List[List[int]] = get_prediction_for_method(dataset, method2)
                assert len(scores1) == len(scores2)
                assert len(pred_list_1) == len(pred_list_2)
                print(len(scores1), len(pred_list_2))
                assert len(scores1) == len(pred_list_2)
                counter = Counter()
                for i in range(len(scores1)):
                    s1 = scores1[i]
                    s2 = scores2[i]
                    p1 = pred_list_1[i]
                    p2 = pred_list_2[i]
                    pred_same = pred_compare(p1, p2)
                    assert s1 is not None
                    assert s2 is not None
                    if pred_same:
                        continue
                    better_fn = get_better_fn(metric)
                    s1_correct = better_fn(0.5, s1)
                    s2_correct = better_fn(0.5, s2)
                    s2_better = better_fn(s1, s2)
                    s1_better = better_fn(s2, s1)
                    if s1_better:
                        decision = "left"
                    elif s2_better:
                        decision = "right"
                    else:
                        decision = "equal"

                    if s1_correct:
                        counter["left correct"] += 1
                    if s2_correct:
                        counter["right correct"] += 1
                    counter[decision] += 1

                total = (counter["left"] + counter["right"] + counter["equal"])
                row = [metric, method1, method2]
                row += [counter[d] / total for d in decisions]
                table.append(row)
        print_table(table)