from typing import List, Tuple

from cache import load_from_pickle
from misc_lib import average
from tab_print import print_table
from tlm.qtype.partial_relevance.qd_segmented_instance import QDSegmentedInstance


def get_all_avg(auc_list: List[Tuple[QDSegmentedInstance, List[float]]]) -> float:
    return average([average(scores) for e, scores in auc_list])


def get_seg_sel_avg(auc_list: List[Tuple[QDSegmentedInstance, List[float]]], q_seg_idx) -> float:
    return average([scores[q_seg_idx] for e, scores in auc_list])


def get_per_label_avg(auc_list: List[Tuple[QDSegmentedInstance, List[float]]], target_label) -> float:
    return average([average(scores) for e, scores in auc_list if e.label == target_label])


def get_per_logit_avg(auc_list: List[Tuple[QDSegmentedInstance, List[float]]], condition_fn) -> float:
    return average([average(scores) for e, scores in auc_list if condition_fn(e.label)])


def main():
    auc_list_list: List[List[Tuple[QDSegmentedInstance, List[float]]]] = load_from_pickle("dev_auc_predict")

    def is_rel(prob):
        return prob >= 0.5

    def is_non_rel(prob):
        return not prob >= 0.5


    for e, scores in auc_list_list[0]:
        print(e.label)

    todo = {
        "All": get_all_avg,
        "Functional": lambda x: get_seg_sel_avg(x, 0),
        "Content": lambda x: get_seg_sel_avg(x, 1),
        "Prediction >= 0.5": lambda x: get_per_logit_avg(x, is_rel),
        "Prediction < 0.5": lambda x: get_per_logit_avg(x, is_non_rel),
    }


    table = []
    for name, metric_fn in todo.items():
        row = [name]
        for method_idx, auc_list in enumerate(auc_list_list):
            score = metric_fn(auc_list)
            row.append(score)
        table.append(row)

    print_table(table)


if __name__ == "__main__":
    main()