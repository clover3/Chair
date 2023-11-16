import random
from typing import NamedTuple, List

import numpy as np

from data_generator2.segmented_enc.es_common.es_two_seg_common import Segment1PartitionedPair, RangePartitionedSegment, \
    IndicesPartitionedSegment, BothSegPartitionedPair
from data_generator2.segmented_enc.es_common.partitioned_encoder import apply_segmentation_to_seg1
from data_generator2.segmented_enc.es_nli.evidence_candidate_gen import pool_delete_indices, pool_sub_seq_indices
from list_lib import lmap
from misc_lib import ceil_divide
from trainer.promise import PromiseKeeper, MyFuture
from trainer_v2.chair_logging import c_log


class PartialSequence(NamedTuple):
    delete_indices: List[int]
    tokens: List[str]


class RandomSeqSelector:
    def __init__(self, g):
        self.g = g

    def get_select_indices(self, segment_2_len):
        g = self.g
        max_sub_seq = 4
        num_del = random.randint(1, max_sub_seq)
        raw_indices: list[int] = pool_sub_seq_indices(num_del, segment_2_len, g)
        indices = list(set(raw_indices))
        indices.sort()
        return indices

    def get_delete_indices(self, segment_2_len):
        sel_indices = self.get_select_indices(segment_2_len)
        output = [i for i in range(segment_2_len) if not i in sel_indices]
        return output


def get_delete_indices_n_gram_selection(segment_2_len, select_size_max):
    i_start = random.randint(0, segment_2_len - select_size_max)
    i_end = min(i_start + select_size_max, segment_2_len)
    indices = list(range(i_start)) + list(range(i_end, segment_2_len))
    return indices


def delete_compute_pep(deleter, pk: PromiseKeeper, tokenizer, max_target_len, pair_data):
    segment2_tokens: list[str] = tokenizer.tokenize(pair_data.segment2)
    pair: Segment1PartitionedPair = apply_segmentation_to_seg1(tokenizer, pair_data)
    segment_2_len = len(segment2_tokens)
    n_sample = 10
    if not isinstance(pair.segment1, RangePartitionedSegment):
        raise TypeError("pair.segment1 should be RangePartitionedSegment")

    segment1: RangePartitionedSegment = pair.segment1
    seg1_part1_len = len(segment1.tokens) - (segment1.ed - segment1.st)
    seg1_part2_len = (segment1.ed - segment1.st)
    c_log.debug(f"Query has {seg1_part1_len} / {seg1_part2_len} tokens", )
    c_log.debug("Label: ", pair_data.label)
    indices_list = [deleter.get_delete_indices(segment_2_len) for _ in range(n_sample)]
    # Adding short evidence case using list comprehension.
    if seg1_part2_len < 6:
        for _ in range(n_sample):
            indices = get_delete_indices_n_gram_selection(segment_2_len, select_size_max=seg1_part2_len)
            indices_list.append(indices)

    indices_list = [indices for indices in indices_list if segment_2_len - len(indices) < max_target_len]

    def get_both_seg_partitioned_pair(indices):
        segment2 = IndicesPartitionedSegment(segment2_tokens, indices, indices)
        c = BothSegPartitionedPair(segment1, segment2, pair_data)
        return c

    candidate_list: List[BothSegPartitionedPair] = lmap(get_both_seg_partitioned_pair, indices_list)
    future_score_list = [pk.get_future(c) for c in candidate_list]
    save_item_per_qd: tuple[list[BothSegPartitionedPair], list[MyFuture[np.array]]] = candidate_list, future_score_list
    return save_item_per_qd