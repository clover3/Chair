from collections import Counter
from typing import List, Dict

from alignment.ists_eval.f1_calc import AlignmentIndexed, dict_bool_check, jaccard_set, get_align_dict
from dataset_specific.ists.parse import AlignmentLabelUnit, AlignmentPredictionList, type_list, ALIGN_NOALI, \
    iSTSProblemWChunk
from list_lib import index_by_fn
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


def overlap(l1, l2):
    for t in l1:
        if t in l2:
            return True
    return False


def get_failure_msg(
        ali_pred: List[AlignmentLabelUnit],
        ali_ref: List[AlignmentLabelUnit],
        ali_gold: List[AlignmentLabelUnit],
        ) -> List:
    # Print when pred is wrong, ref is correct
    def simplify_type(type_l):
        for s in type_list:
            if s in type_l:
                return s

    wrong_indices = get_wrong_indices(ali_gold, ali_pred, simplify_type)
    wrong_indices_ref = get_wrong_indices(ali_gold, ali_ref, simplify_type)


    msg_list = []
    for align_unit in ali_pred:
        if overlap(align_unit.chunk_token_id1, wrong_indices) \
                and not overlap(align_unit.chunk_token_id1, wrong_indices_ref):
            gold_align = get_corresponding_align_unit(ali_gold, align_unit)
            ref_align = get_corresponding_align_unit(ali_ref, align_unit)

            left_type = simplify_type(align_unit.align_types)
            if gold_align is not None:
                ref_type = simplify_type(ref_align.align_types)
                gold_type = simplify_type(gold_align.align_types)
                msg = f"{left_type} -> {ref_type} -> {gold_type}: {align_unit.chunk_text1}, {align_unit.chunk_text2}"
                if align_unit.chunk_text1 == gold_align.chunk_text1 and \
                        align_unit.chunk_text2 == gold_align.chunk_text2:
                    pass
                else:
                    msg += f" -> {gold_align.chunk_text1}, {gold_align.chunk_text2}"
            else:
                msg = f"{left_type} -> NOALI: {align_unit.chunk_text1}, {align_unit.chunk_text2}"
            msg_list.append(msg)
    return msg_list


def get_corresponding_align_unit(ali_gold, align_unit):
    gold_align = None
    for align_unit_gold in ali_gold:
        if overlap(align_unit_gold.chunk_token_id1, align_unit.chunk_token_id1):
            gold_align = align_unit_gold
            break
    return gold_align


def get_wrong_indices(ali_gold, ali_pred, simplify_type):
    align1 = AlignmentIndexed.from_alignment_list(ali_pred)
    align2 = AlignmentIndexed.from_alignment_list(ali_gold)
    wrong_indices = []
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
            correct = left_type == right_type
            if not correct:
                wrong_indices.append(t1)
    return wrong_indices


def print_failure_outer(
        gold: AlignmentPredictionList,
        pred: AlignmentPredictionList,
        pred2: AlignmentPredictionList,
        problems: List[iSTSProblemWChunk],
        ):
    gold_d = dict(gold)
    pred2_d = dict(pred2)

    problems_d = index_by_fn(lambda x: x.problem_id, problems)
    for problem_id, pred_align in pred:

        msg_list = get_failure_msg(
            pred_align,
            pred2_d[problem_id],
            gold_d[problem_id]
        )
        problem = problems_d[problem_id]
        if msg_list:
            print(problem_id)
            print(problem.text1)
            print(problem.text2)
            for l in msg_list:
                print(l)
