from collections import defaultdict
from typing import List, Dict

from alignment.ists_eval.eval_helper import load_ists_predictions
from dataset_specific.ists.parse import AlignmentLabelUnit, AlignmentPredictionList
from dataset_specific.ists.path_helper import load_ists_label
from misc_lib import get_f1


def count_fanout(sub_align: list[AlignmentLabelUnit]) -> float:
    edge12, edge21 = get_align_dict(sub_align)
    count = 0
    # Assume 2-dimensional array where m[t1,t2] indicates the alignment prediction
    # Each m[t1,t2] is weighted by s, which is number the of tokens aligned for t1 or t2

    # If t1 is not aligned, it will be aligned to 0,
    for t1 in edge12:
        for t2 in edge12[t1]:
            s1 = len(edge12[t1])
            s2 = len(edge21[t2])
            s = max(s1, s2)
            count += 1 / s

    return count


class AlignmentIndexed:
    def __init__(self, edge12, edge21, alu_list):
        self.edge12 = edge12
        self.edge21 = edge21
        self.alu_list = alu_list

    @classmethod
    def from_alignment_list(cls, l: List[AlignmentLabelUnit]):
        edge12, edge21 = get_align_dict(l)
        return AlignmentIndexed(edge12, edge21, l)

    @classmethod
    def score_d_from_alignment_list(cls, l: List[AlignmentLabelUnit]):
        edge12, edge21 = get_align_score_dict(l)
        return AlignmentIndexed(edge12, edge21, l)


def jaccard_set(l1: List[str], l2: List[str]):
    def convert(l):
        return set([s.lower().strip() for s in l])

    s1, s2 = convert(l1), convert(l2)
    output = len(s1.intersection(s2)) / len(s1.union(s2))
    return output


def get_align_dict(sub_align: List[AlignmentLabelUnit]):
    edge12: Dict[int, Dict[int, List[str]]] = defaultdict(dict)
    edge21: Dict[int, Dict[int, List[str]]] = defaultdict(dict)
    for align in sub_align:
        tokens1 = align.chunk_text1.split()
        tokens2 = align.chunk_text2.split()
        for j1, t1 in enumerate(align.chunk_token_id1):
            if t1 == 0:
                continue
            if tokens1[j1] in ".,:\'\`?;\"-":
                continue
            for j2, t2 in enumerate(align.chunk_token_id2):
                if t2 == 0:
                    continue
                if tokens2[j2] in ".,:\'\`?;\"-":
                    continue
                edge12[t1][t2] = align.align_types
                edge21[t2][t1] = align.align_types
    return edge12, edge21


def get_align_score_dict(sub_align: List[AlignmentLabelUnit]):
    edge12: Dict[int, Dict[int, int]] = defaultdict(dict)
    edge21: Dict[int, Dict[int, int]] = defaultdict(dict)
    for align in sub_align:
        tokens1 = align.chunk_text1.split()
        tokens2 = align.chunk_text2.split()
        for j1, t1 in enumerate(align.chunk_token_id1):
            if t1 == 0:
                continue
            if tokens1[j1] in ".,:\'\`?;\"-":
                continue
            for j2, t2 in enumerate(align.chunk_token_id2):
                if t2 == 0:
                    continue
                if tokens2[j2] in ".,:\'\`?;\"-":
                    continue
                edge12[t1][t2] = int(align.align_score)
                edge21[t2][t1] = int(align.align_score)
    return edge12, edge21


def dict_bool_check(align2, t1, t2):
    try:
        f_match = bool(align2.edge12[t1][t2])
    except KeyError:
        f_match = False
    return f_match


def _calc_overlap_w_score(ali1: List[AlignmentLabelUnit],
                 ali2: List[AlignmentLabelUnit], mode="") -> float:
    n_overlap = 0
    align1 = AlignmentIndexed.from_alignment_list(ali1)
    align2 = AlignmentIndexed.from_alignment_list(ali2)

    align_score1 = AlignmentIndexed.score_d_from_alignment_list(ali1)
    align_score2 = AlignmentIndexed.score_d_from_alignment_list(ali2)

    for t1 in align1.edge12:
        for t2 in align1.edge12[t1]:
            f_match = dict_bool_check(align2, t1, t2)
            if f_match:
                if mode == "score":
                    factor = get_score_error_factor(align_score1, align_score2, t1, t2)
                elif mode == "typescore":
                    raise NotImplementedError
                    # Penality is removed when score error is small
                    factor1 = jaccard_set(align1.edge12[t1][t2], align2.edge12[t1][t2])
                    factor2 = get_score_error_factor(align_score1, align_score2, t1, t2)
                    factor = factor1 * factor2
                else:
                    raise NotImplementedError

                s1 = len(align1.edge12[t1])
                s2 = len(align1.edge21[t2])
                s = max(s1, s2)
                n_overlap += 1 / s * factor

    return n_overlap


def get_score_error_factor(align_score1, align_score2, t1, t2):
    s1 = align_score1.edge12[t1][t2]
    s2 = align_score2.edge12[t1][t2]
    factor = 1 - abs(s1 - s2) / 5
    return factor


def _calc_overlap_no_score(ali1: List[AlignmentLabelUnit],
                          ali2: List[AlignmentLabelUnit], mode="") -> float:
    n_overlap = 0
    align1 = AlignmentIndexed.from_alignment_list(ali1)
    align2 = AlignmentIndexed.from_alignment_list(ali2)

    for t1 in align1.edge12:
        for t2 in align1.edge12[t1]:
            f_match = dict_bool_check(align2, t1, t2)
            if f_match:
                if mode == "":
                    factor = 1
                elif mode == "type":
                    factor = jaccard_set(align1.edge12[t1][t2], align2.edge12[t1][t2])
                else:
                    raise NotImplementedError

                s1 = len(align1.edge12[t1])
                s2 = len(align1.edge21[t2])
                s = max(s1, s2)
                n_overlap += 1 / s * factor

    return n_overlap


def calc_overlap(ali1: List[AlignmentLabelUnit],
                 ali2: List[AlignmentLabelUnit], mode="") -> float:
    if "score" in mode:
        return _calc_overlap_w_score(ali1, ali2, mode)
    else:
        return _calc_overlap_no_score(ali1, ali2, mode)


def count_fanout_sum(alignment_list: AlignmentPredictionList):
    link_sys = 0
    for problem_id, sub_align in alignment_list:
        fanout = count_fanout(sub_align)
        link_sys += fanout
    return link_sys


def calc_f1(gold: AlignmentPredictionList,
            pred: AlignmentPredictionList,
            mode) -> Dict[str, float]:
    gold_d = dict(gold)
    pred_d = dict(pred)
    if len(gold) != len(pred):
        print("Gold has {} problems but pred has only {} problems".format(len(gold), len(pred)))
        gold = [(problem_id, alignments) for problem_id, alignments in gold if problem_id in pred_d]
        print("Adjusting to predictions")
    link_pred = count_fanout_sum(pred)
    link_gold = count_fanout_sum(gold)

    overlap_pred = 0
    overlap_gold = 0
    for problem_id, pred_align in pred:
        n_overlap = calc_overlap(pred_align, gold_d[problem_id], mode)
        overlap_pred += n_overlap
        total = count_fanout(pred_align)
        prec = n_overlap / total if total else 0
        # print("{} ov={} prec: {}".format(problem_id, n_overlap, two_digit_float(prec)))
        overlap_gold += calc_overlap(gold_d[problem_id], pred_align, mode)

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


def main():
    pred: AlignmentPredictionList = load_ists_predictions("headlines", "train", "em")
    gold: AlignmentPredictionList = load_ists_label("headlines", "train")
    for mode in ["", "type", "score"]:
        scores = calc_f1(gold, pred, mode)
        print(scores)


if __name__ == "__main__":
    main()