from typing import List

import numpy as np

from data_generator2.segmented_enc.es_nli.common import HSegmentedPair, PHSegmentedPair

SegmentedPair2 = PHSegmentedPair

def get_delete_indices(attn_merged, e: HSegmentedPair) -> List[List[int]]:
    """
    Select indices to delete, in the order of smaller attention weights.
    :param attn_merged: np.array, [seq_len, seq_len]
    :param e: Segmented Piar
    :return: List of indices to delete
    """
    tokens_all = ["[CLS]"] + e.p_tokens + ["[SEP]"] + e.h_tokens + ["[SEP]"]
    p_st = 1
    p_ed = p_st + len(e.p_tokens)
    h_st = p_ed + 1
    h_split_st = h_st + e.st
    h_split_ed = h_st + e.ed
    h_ed = h_st + len(e.h_tokens)
    h_part1_from = np.concatenate([attn_merged[p_st:p_ed, h_st: h_split_st],
                                   attn_merged[p_st:p_ed, h_split_ed: h_ed]],
                                  axis=1)
    h_part1_to = np.concatenate([attn_merged[h_st: h_split_st, p_st:p_ed],
                                 attn_merged[h_split_ed: h_ed, p_st:p_ed]],
                                axis=0)
    h_part2_from = attn_merged[p_st:p_ed, h_split_st: h_split_ed]
    h_part2_to = attn_merged[h_split_st: h_split_ed, p_st:p_ed]
    h_part1_from_mean = np.mean(h_part1_from, axis=1)
    h_part2_from_mean = np.mean(h_part2_from, axis=1)
    h_part1_to_mean = np.mean(h_part1_to, axis=0)
    h_part2_to_mean = np.mean(h_part2_to, axis=0)
    h_part_i_from_mean = [h_part1_from_mean, h_part2_from_mean]
    h_part_i_to_mean = [h_part1_to_mean, h_part2_to_mean]

    second_len = e.ed - e.st
    h_part_len = [len(e.h_tokens) - second_len, second_len]
    h_tokens_i = [e.get_first_h_tokens_w_mask(), e.get_second_h_tokens()]
    delete_indices_list: List[List[int]] = []
    for i in [0, 1]:
        # assert len(h_part_i_from_mean[i]) == len(h_part_i_to_mean[i])

        h_part_i_mean = (h_part_i_from_mean[i] + h_part_i_to_mean[i]) / 2
        # assert len(h_part_i_mean) == len(e.p_tokens)

        other_part_len = h_part_len[1 - i]
        n_delete = other_part_len
        n_max_delete = len(e.p_tokens) - len(h_tokens_i[i])
        if n_delete > n_max_delete > 1:
            n_delete = n_max_delete

        delete_indices = list(np.argsort(h_part_i_mean)[:n_delete])
        delete_indices_list.append(delete_indices)
    return delete_indices_list
