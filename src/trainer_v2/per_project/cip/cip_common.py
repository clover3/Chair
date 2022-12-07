import random
from collections import defaultdict, Counter
from typing import Tuple, NamedTuple, List

import numpy as np

from cache import named_tuple_to_json
from data_generator.special_tokens import MASK_ID
from misc_lib import SuccessCounter
from tab_print import print_table
from trainer.promise import MyFuture, list_future


def get_random_split_location(ids) -> Tuple[int, int]:
    if len(ids) > 1:
        st = random.randint(0, len(ids) - 2)
        ed = random.randint(st + 1, len(ids) - 1)
    else:
        st = 0
        ed = 0
    return st, ed


def split_into_two(hypo, st, ed):
    first_a = hypo[:st]
    first_b = hypo[ed:]
    hypo1 = first_a + [MASK_ID] + first_b
    hypo2 = hypo[st:ed]
    return hypo1, hypo2


class SegmentationTrialInputs(NamedTuple):
    prem: List[int]
    hypo: List[int]
    label: int
    nlits_input_future_lists: List[MyFuture]
    ts_input_info_list: List[Tuple[int, int]]


class ComparisonF(NamedTuple):
    prem: List[int]
    hypo: List[int]
    label: int
    full_input_future: MyFuture
    nlits_input_future_lists: List[MyFuture]
    ts_input_info_list: List[Tuple[int, int]]


class Comparison(NamedTuple):
    prem: List[int]
    hypo: List[int]
    label: int
    full_pred_probs: List[float]
    ts_pred_probs: List[List[float]]
    ts_input_info_list: List[Tuple[int, int]]

    @classmethod
    def from_comparison_f(cls, cf: ComparisonF):
        full_probs = cf.full_input_future.get()
        ts_pred_list = list_future(cf.nlits_input_future_lists)
        return Comparison(cf.prem, cf.hypo, cf.label, full_probs, ts_pred_list, cf.ts_input_info_list)


class SegmentationTrials(NamedTuple):
    prem: List[int]
    hypo: List[int]
    label: int
    seg_outputs: List
    st_ed_list: List[Tuple[int, int]]

    @classmethod
    def from_sti(cls, si: SegmentationTrialInputs):
        ts_pred_list = list_future(si.nlits_input_future_lists)
        return SegmentationTrials(si.prem, si.hypo, si.label, ts_pred_list, si.ts_input_info_list)

    def to_json(self):
        return named_tuple_to_json(self)

    @classmethod
    def from_json(cls, j):
        return SegmentationTrials(j['prem'], j['hypo'], j['label'],
                                  j['seg_outputs'], j['st_ed_list'])


class Prediction(NamedTuple):
    prem: List[int]
    hypo: List[int]
    label: int
    pred: List[float]

    def to_json(self):
        return named_tuple_to_json(self)

    @classmethod
    def from_json(cls, j):
        return Prediction(j['prem'], j['hypo'], j['label'], j['pred'])


def get_statistics(iterate_comparison):
    confusion = Counter()
    suc_counters = defaultdict(SuccessCounter)
    for comparison in iterate_comparison:
        full_pred = np.argmax(comparison.full_pred_probs)
        any_bad_seg = False
        for ts_probs, info in zip(comparison.ts_pred_probs, comparison.ts_input_info_list):
            ts_probs: Tuple[List[List[float]], List[float]] = ts_probs
            local_d, global_d = ts_probs
            seg1_pred = local_d[0]
            seg2_pred = local_d[1]
            ts_global_pred = np.argmax(global_d)
            if full_pred != ts_global_pred and full_pred == comparison.label:
                suc_counters['bad_seg_rate'].suc()
                confusion[(full_pred, ts_global_pred)] += 1
                any_bad_seg = True
            else:
                suc_counters['bad_seg_rate'].fail()

            suc_counters['ts_acc'].add(ts_global_pred == comparison.label)
        suc_counters['bad_seg_exists'].add(any_bad_seg)
        suc_counters['full_acc'].add(full_pred == comparison.label)
    for key, suc_counter in suc_counters.items():
        print(key, suc_counter.get_suc_prob())

    table = []
    for i in range(3):
        row = []
        for j in range(3):
            row.append(confusion[i,j])
        table.append(row)
    print_table(table)