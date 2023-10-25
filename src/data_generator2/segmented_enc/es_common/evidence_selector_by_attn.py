from typing import List, Callable

import numpy as np

from data_generator2.segmented_enc.es_common.es_two_seg_common import Segment2PartitionedPair, RangePartitionedSegment, \
    Segment1PartitionedPair, BothSegPartitionedPair, IndicesPartitionedSegment, PartitionedSegment


def get_delete_indices_for_segment1(attn_merged, e: Segment2PartitionedPair, get_num_delete) -> List[List[int]]:
    """
    Select indices to delete, in the order of smaller attention weights.
    :param attn_merged: np.array, [seq_len, seq_len]
    :param e: Segmented Piar
    :return: List of indices to delete
    """
    seg1_st = 1
    seg1_ed = seg1_st + len(e.segment2.tokens)
    seg2_st = seg1_ed + 1
    seg2_ed = seg2_st + len(e.segment1)

    # part_segment: The segment that is PARTitioned
    # cont_seg: The segment that is NOT partitioned, and thus CONTinuous
    assert isinstance(e.segment2, RangePartitionedSegment)
    part_segment: RangePartitionedSegment = e.segment2
    cont_seg_tokens = e.segment1
    part_seg_st = seg1_st
    part_seg_ed = seg1_ed
    cont_seg_st = seg2_st
    cont_seg_ed = seg2_ed

    return get_delete_indices_inner(
        attn_merged,
        cont_seg_tokens, cont_seg_st, cont_seg_ed,
        part_segment, part_seg_st, part_seg_ed,
        get_num_delete)


def get_delete_indices_for_segment2(attn_merged, e: Segment1PartitionedPair, get_num_delete) -> List[List[int]]:
    """
    Select indices to delete, in the order of smaller attention weights.
    :param attn_merged: np.array, [seq_len, seq_len]
    :param e: Segmented Piar
    :return: List of indices to delete
    """
    segment1 = e.segment1
    segment2 = e.segment2
    return get_delete_indices_for_segment2_inner(attn_merged, segment1, segment2, get_num_delete)


def get_delete_indices_for_segment2_inner(
        attn_merged,
        segment1: PartitionedSegment,
        segment2: List[str],
        get_num_delete):
    seg1_st = 1
    seg1_ed = seg1_st + len(segment1.tokens)
    seg2_st = seg1_ed + 1
    seg2_ed = seg2_st + len(segment2)
    # part_segment: The segment that is PARTitioned
    # cont_seg: The segment that is NOT partitioned, and thus CONTinuous
    assert isinstance(segment1, RangePartitionedSegment)
    part_segment: RangePartitionedSegment = segment1
    cont_seg_tokens = segment2
    part_seg_st = seg1_st
    part_seg_ed = seg1_ed
    cont_seg_st = seg2_st
    cont_seg_ed = seg2_ed
    return get_delete_indices_inner(
        attn_merged,
        cont_seg_tokens, cont_seg_st, cont_seg_ed,
        part_segment, part_seg_st, part_seg_ed,
        get_num_delete)


def get_delete_indices_inner(
        attn_merged: np.array,
        cont_seg_tokens: List[str], cont_seg_st: int, cont_seg_ed: int,
        part_segment: RangePartitionedSegment, part_seg_st: int, part_seg_ed: int,
        get_num_delete: Callable[[int, int, int], int]) -> List[List[int]]:
    """

    :param attn_merged: np.array, [seq_len, seq_len]
    :param cont_seg_tokens:
    :param cont_seg_st: Starting location of continuous segment in the concatenated sequence
    :param cont_seg_ed: Ending location of continuous segment in the concatenated sequence
    :param part_segment:
    :param part_seg_st: Starting location of partitioned segment in the concatenated sequence
    :param part_seg_ed: Ending location of partitioned segment in the concatenated sequence
    :param get_num_delete: Function that returns how many tokens to delete, cur_part_len, other_part_len, len(cont_seg_tokens)
    :return: List of indice to delete (two)
    """
    part_seg_part_i_from_mean, part_seg_part_i_to_mean = merge_attn_scores_for_partitions(attn_merged, cont_seg_st, cont_seg_ed,
                                                                                          part_segment, part_seg_st, part_seg_ed)
    second_len = part_segment.ed - part_segment.st
    part_seg_part_len = [len(part_segment.tokens) - second_len, second_len]
    delete_indices_list: List[List[int]] = []
    for i in [0, 1]:
        part_seg_part_i_mean = (part_seg_part_i_from_mean[i] + part_seg_part_i_to_mean[i]) / 2

        cur_part_len = part_seg_part_len[i]
        other_part_len = part_seg_part_len[1 - i]
        n_delete = get_num_delete(cur_part_len, other_part_len, len(cont_seg_tokens))
        delete_indices = list(np.argsort(part_seg_part_i_mean)[:n_delete])
        delete_indices_list.append(delete_indices)
    return delete_indices_list


def merge_attn_scores_for_partitions(attn_merged, cont_seg_st, cont_seg_ed, part_segment, part_seg_st, part_seg_ed):
    part_seg_split_st = part_seg_st + part_segment.st
    part_seg_split_ed = part_seg_st + part_segment.ed
    part_seg_part1_from = np.concatenate(
        [attn_merged[cont_seg_st:cont_seg_ed, part_seg_st: part_seg_split_st],
         attn_merged[cont_seg_st:cont_seg_ed, part_seg_split_ed: part_seg_ed]],
        axis=1)
    part_seg_part1_to = np.concatenate(
        [attn_merged[part_seg_st: part_seg_split_st, cont_seg_st:cont_seg_ed],
         attn_merged[part_seg_split_ed: part_seg_ed, cont_seg_st:cont_seg_ed]],
        axis=0)
    part_seg_part2_from = attn_merged[cont_seg_st:cont_seg_ed, part_seg_split_st: part_seg_split_ed]
    part_seg_part2_to = attn_merged[part_seg_split_st: part_seg_split_ed, cont_seg_st:cont_seg_ed]
    part_seg_part1_from_mean = np.mean(part_seg_part1_from, axis=1)
    part_seg_part2_from_mean = np.mean(part_seg_part2_from, axis=1)
    part_seg_part1_to_mean = np.mean(part_seg_part1_to, axis=0)
    part_seg_part2_to_mean = np.mean(part_seg_part2_to, axis=0)
    part_seg_part_i_from_mean = [part_seg_part1_from_mean, part_seg_part2_from_mean]
    part_seg_part_i_to_mean = [part_seg_part1_to_mean, part_seg_part2_to_mean]
    return part_seg_part_i_from_mean, part_seg_part_i_to_mean


def compute_attn_sel_delete_indices(
        e: Segment1PartitionedPair,
        attn_score: np.array,
        get_num_delete
        ) -> BothSegPartitionedPair:
    delete_indices_list = get_delete_indices_for_segment2(attn_score, e, get_num_delete)
    segment2 = IndicesPartitionedSegment(e.segment2, delete_indices_list[0], delete_indices_list[1])
    e_out = BothSegPartitionedPair(e.segment1, segment2, e.pair_data)
    return e_out