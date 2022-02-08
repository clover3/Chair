import numpy as np

from data_generator.bert_input_splitter import get_sep_loc
from tlm.qtype.partial_relevance.eval_data_structure import SegmentedInstance


def get_always_active_mask(seg_inst: SegmentedInstance, max_seq_length) -> np.array:
    #       [CLS] [Seg1] [SEP] [Seg2] [SEP]
    # [CLS]   1     1      1     1      1
    # [Seg1]  1     1      1     0      1
    # [SEP]   1     1      1     1      1
    # [Seg2]  1     0      1     1      1
    # [SEP]   1     1      1     1      1
    l1 = len(seg_inst.text1.tokens_ids)
    l2 = len(seg_inst.text2.tokens_ids)
    total_l = l1 + l2 + 3

    mask = get_always_active_mask_inner(max_seq_length, l1, l2, total_l)

    return mask


def get_always_active_mask_inner(max_seq_length, l1, l2, total_l):
    def is_seg1(i):
        return 1 <= i < 1 + l1

    def is_seg2(i):
        return 2 + l1 <= i < 2 + l1 + l2

    mask = np.zeros([max_seq_length, max_seq_length])
    for i1 in range(max_seq_length):
        for i2 in range(max_seq_length):
            if is_seg1(i1) and is_seg2(i2):
                mask[i1, i2] = 0
            elif is_seg2(i1) and is_seg1(i2):
                mask[i1, i2] = 0
            elif i1 < total_l and i2 < total_l:
                mask[i1, i2] = 1
            else:
                mask[i1, i2] = 0
    return mask


def get_always_active_mask_w_input_ids(input_ids):
    max_seq_length = len(input_ids)
    first_sep_loc, second_sep_loc = get_sep_loc(input_ids)
    l1 = first_sep_loc - 1
    l2 = (second_sep_loc - first_sep_loc) - 1
    total_l = l1 + l2 + 3

    def is_seg1(i):
        return 1 <= i < 1 + l1

    def is_seg2(i):
        return 2 + l1 <= i < 2 + l1 + l2

    mask = np.zeros([max_seq_length, max_seq_length])
    for i1 in range(max_seq_length):
        for i2 in range(max_seq_length):
            if is_seg1(i1) and is_seg2(i2):
                mask[i1, i2] = 0
            elif is_seg2(i1) and is_seg1(i2):
                mask[i1, i2] = 0
            elif i1 < total_l and i2 < total_l:
                mask[i1, i2] = 1
            else:
                mask[i1, i2] = 0
    return mask


def get_always_active_mask_qt_inner(max_seq_length, l1, l2, total_l, always_on):
    seg1_shifted_non_target = [i + 1 for i in always_on]

    def is_seg1_target(i):
        return 1 <= i < 1 + l1 and i not in seg1_shifted_non_target

    def is_seg1_non_target(i):
        return i in seg1_shifted_non_target

    def is_seg2(i):
        return 2 + l1 <= i < 2 + l1 + l2

    mask = np.zeros([max_seq_length, max_seq_length])
    for i1 in range(max_seq_length):
        for i2 in range(max_seq_length):
            if is_seg1_target(i1) and is_seg2(i2):
                mask[i1, i2] = 0
            elif is_seg2(i1) and is_seg1_target(i2):
                mask[i1, i2] = 0
            elif i1 < total_l and i2 < total_l:
                mask[i1, i2] = 1
            else:
                mask[i1, i2] = 0
    return mask


def get_always_active_mask_qt(seg_inst: SegmentedInstance,
                              max_seq_length,
                              target_idx) -> np.array:
    #       [CLS] [Seg1] [SEP] [Seg2] [SEP]
    # [CLS]   1     1      1     1      1
    # [Seg1]  1     1      1     0      1
    # [SEP]   1     1      1     1      1
    # [Seg2]  1     0      1     1      1
    # [SEP]   1     1      1     1      1
    l1 = len(seg_inst.text1.tokens_ids)
    l2 = len(seg_inst.text2.tokens_ids)
    total_l = l1 + l2 + 3

    always_on_indices = []
    for seg_idx in seg_inst.text1.enum_seg_idx():
        if seg_idx != target_idx:
            always_on_indices.extend(seg_inst.text1.seg_token_indices[seg_idx])
    mask = get_always_active_mask_qt_inner(max_seq_length, l1, l2, total_l, always_on_indices)

    return mask
