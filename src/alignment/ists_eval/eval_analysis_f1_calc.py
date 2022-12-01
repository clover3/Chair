from collections import Counter
from typing import List, Dict

from alignment.ists_eval.f1_calc import AlignmentIndexed, dict_bool_check, jaccard_set, get_align_dict
from dataset_specific.ists.parse import AlignmentLabelUnit, AlignmentPredictionList, type_list, ALIGN_NOALI
from misc_lib import get_f1


def calc_overlap_type_specific(
        ali1: List[AlignmentLabelUnit],
        ali2: List[AlignmentLabelUnit],
        target_type) -> float:
    n_overlap = 0
    align1 = AlignmentIndexed.from_alignment_list(ali1)
    align2 = AlignmentIndexed.from_alignment_list(ali2)

    for t1 in align1.edge12:
        for t2 in align1.edge12[t1]:
            f_match = dict_bool_check(align2, t1, t2)
            left_type = align1.edge12[t1][t2]
            if f_match:
                right_type = align2.edge12[t1][t2]
                factor = jaccard_set(left_type, right_type)
                if target_type in left_type:
                    s1 = len(align1.edge12[t1])
                    s2 = len(align1.edge21[t2])
                    s = max(s1, s2)
                    n_overlap += 1 / s * factor
    return n_overlap


def count_fanout_type_specific(
        sub_align: list[AlignmentLabelUnit],
        target_label) -> float:
    edge12, edge21 = get_align_dict(sub_align)
    count = 0
    # Assume 2-dimensional array where m[t1,t2] indicates the alignment prediction
    # Each m[t1,t2] is weighted by s, which is number the of tokens aligned for t1 or t2
    # If t1 is not aligned, it will be aligned to 0,
    for t1 in edge12:
        for t2 in edge12[t1]:
            label = edge12[t1][t2]
            if target_label in label:
                s1 = len(edge12[t1])
                s2 = len(edge21[t2])
                s = max(s1, s2)
                count += 1 / s

    return count


def count_fanout_sum_type_specific(
        alignment_list: AlignmentPredictionList,
        target_label):
    link_sys = 0
    for problem_id, sub_align in alignment_list:
        fanout = count_fanout_type_specific(sub_align, target_label)
        link_sys += fanout
    return link_sys


def calc_type_specific_f1(
        gold: AlignmentPredictionList,
        pred: AlignmentPredictionList,
        target_label) -> Dict[str, float]:
    gold_d = dict(gold)
    pred_d = dict(pred)
    if len(gold) != len(pred):
        print("Gold has {} problems but pred has only {} problems".format(len(gold), len(pred)))
        gold = [(problem_id, alignments) for problem_id, alignments in gold if problem_id in pred_d]
        print("Adjusting to predictions")
    link_pred = count_fanout_sum_type_specific(pred, target_label)
    link_gold = count_fanout_sum_type_specific(gold, target_label)

    overlap_pred = 0
    overlap_gold = 0
    for problem_id, pred_align in pred:
        overlap_pred += calc_overlap_type_specific(pred_align, gold_d[problem_id], target_label)
        overlap_gold += calc_overlap_type_specific(gold_d[problem_id], pred_align, target_label)

    precision = overlap_pred / link_pred if link_pred != 0 else 1
    recall = overlap_gold / link_gold if link_gold !=0 else 1
    f1 = get_f1(precision, recall)
    return {
        'overlap_pred': overlap_pred,
        'overlap_gold': overlap_gold,
        'link_pred': link_pred,
        'link_gold': link_gold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def confusion_count(
        ali1: List[AlignmentLabelUnit],
        ali2: List[AlignmentLabelUnit],
        ) -> Counter:
    def simplify_type(type_l):
        for s in type_list:
            if s in type_l:
                return s
    align1 = AlignmentIndexed.from_alignment_list(ali1)
    align2 = AlignmentIndexed.from_alignment_list(ali2)
    counter = Counter()
    for t1 in align1.edge12:
        for t2 in align1.edge12[t1]:
            f_match = dict_bool_check(align2, t1, t2)
            left_type = simplify_type(align1.edge12[t1][t2])
            if f_match:
                right_type = simplify_type(align2.edge12[t1][t2])
            else:
                right_type = ALIGN_NOALI
            s1 = len(align1.edge12[t1])
            s2 = len(align1.edge21[t2])
            s = max(s1, s2)
            counter[(left_type, right_type)] += 1 / s
    return counter


def calc_confucion(
        gold: AlignmentPredictionList,
        pred: AlignmentPredictionList,
        ):
    gold_d = dict(gold)
    pred_d = dict(pred)

    conf_counter_pg = Counter()
    conf_counter_gp = Counter()

    for problem_id, pred_align in pred:
        conf_counter_pg.update(confusion_count(pred_align, gold_d[problem_id]))
        conf_counter_gp.update(confusion_count(gold_d[problem_id], pred_align))

    return conf_counter_pg, conf_counter_gp