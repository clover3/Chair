from typing import NamedTuple, List

import numpy as np

from dataset_specific.mnli.mnli_reader import NLIPairData
from list_lib import list_equal


class SegmentedPair(NamedTuple):
    p_tokens: List[str]
    h_tokens: List[str]
    st: int
    ed: int
    nli_pair: NLIPairData

    def get_first_h_tokens(self):
        return self.h_tokens[:self.st], self.h_tokens[self.ed:]

    def get_first_h_tokens_w_mask(self):
        return self.h_tokens[:self.st] + ["[MASK]"] + self.h_tokens[self.ed:]

    def get_second_h_tokens(self):
        return self.h_tokens[self.st: self.ed]


class SegmentedPair2(NamedTuple):
    p_tokens: List[str]
    h_tokens: List[str]
    h_st: int
    h_ed: int
    p_del_indices1: List[int]
    p_del_indices2: List[int]
    nli_pair: NLIPairData

    def get_partial_prem(self, segment_idx: int) -> List[str]:
        assert segment_idx == 0 or segment_idx == 1

        p_tokens_new = list(self.p_tokens)
        del_indices = [self.p_del_indices1, self.p_del_indices2][segment_idx]
        for i in del_indices:
            p_tokens_new[i] = "[MASK]"
        return p_tokens_new

    def get_partial_hypo(self, segment_idx: int) -> List[str]:
        if segment_idx == 0:
            return self.h_tokens[:self.h_st] + ["[MASK]"] + self.h_tokens[self.h_ed:]
        elif segment_idx == 1:
            return self.h_tokens[self.h_st: self.h_ed]
        else:
            raise Exception()



def get_delete_indices(attn_merged, e: SegmentedPair) -> List[List[int]]:
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
    # assert list_equal(tokens_all[p_st: p_ed], e.p_tokens)
    # assert list_equal(tokens_all[h_st: h_ed], e.h_tokens)
    # assert list_equal(tokens_all[h_st: h_split_st], e.h_tokens[:e.st])
    # assert list_equal(tokens_all[h_split_st: h_split_ed], e.h_tokens[e.st: e.ed])
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
