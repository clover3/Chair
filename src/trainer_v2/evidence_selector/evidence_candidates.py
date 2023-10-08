from typing import NamedTuple, List

import numpy as np

from data_generator.tokenizer_wo_tf import get_tokenizer


class EvidencePair(NamedTuple):
    p_tokens: List[int]
    h1: List[int]
    h2: List[int]
    p_del_indices1: List[int]
    p_del_indices2: List[int]

    def is_base_inst(self):
        return len(self.p_del_indices1) == 0 and len(self.p_del_indices2) == 0


def get_st_ed(segment_ids):
    seg2_start = None
    seg2_end = None
    for i, t in enumerate(segment_ids):
        if segment_ids[i] == 1 and seg2_start is None:
            seg2_start = i
        elif segment_ids[i] == 0 and seg2_start is not None:
            seg2_end = i
            break
    if seg2_start is None:
        seg2_start = len(segment_ids)
    if seg2_end is None:
        seg2_end = len(segment_ids)
    return seg2_start, seg2_end


class PHSegmentedPairParser:
    def __init__(self, segment_len):
        self.segment_len = segment_len
        self.tokenizer = get_tokenizer()
        self.mask_id = self.tokenizer.wordpiece_tokenizer.vocab["[MASK]"]

    def get_ph_segment_pair(self, input_ids, segment_ids) -> EvidencePair:
        input_ids = input_ids.numpy().tolist()
        segment_ids = segment_ids.numpy().tolist()

        input_ids1 = input_ids[:self.segment_len]
        segment_ids1 = segment_ids[:self.segment_len]

        input_ids2 = input_ids[self.segment_len:]
        segment_ids2 = segment_ids[self.segment_len:]

        def split(input_ids, segment_ids):
            seg2_start, seg2_end = get_st_ed(segment_ids)
            seg1 = input_ids[1: seg2_start-1]
            seg2 = input_ids[seg2_start: seg2_end-1]
            return seg1, seg2

        p1, h1 = split(input_ids1, segment_ids1)
        p2, h2 = split(input_ids2, segment_ids2)

        assert len(p1) == len(p2)

        p_merged = []
        for i, t in enumerate(p1):
            if t == self.mask_id:
                p_merged.append(p2[i])
            else:
                p_merged.append(p1[i])

        def get_del_indices(p_i):
            indices = []
            for i, t in enumerate(p_i):
                if t == self.mask_id:
                    indices.append(i)
            return indices

        return EvidencePair(p_merged, h1, h2, get_del_indices(p1), get_del_indices(p2))


class ScoredEvidencePair(NamedTuple):
    pair: EvidencePair
    g_y: np.array
    l_y: np.array





