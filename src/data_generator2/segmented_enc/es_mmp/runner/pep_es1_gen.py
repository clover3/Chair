import random

import numpy as np
import sys
from typing import List, Iterable, Callable, Dict, Tuple, Set

from transformers import AutoTokenizer

from data_generator2.segmented_enc.es_common.es_two_seg_common import BothSegPartitionedPair, Segment1PartitionedPair, \
    MaskPartitionedSegment, RangePartitionedSegment
from data_generator2.segmented_enc.es_common.partitioned_encoder import PartitionedEncoderIF, PartitionedEncoder
from data_generator2.segmented_enc.es_mmp.pep_es_common import generate_train_data, PairWithESEncoderIF, QueryDocES
from list_lib import list_equal, assert_list_equal
from taskman_client.wrapper3 import JobContext

# Delete strategy
#    1. Delete indices
#    2. Keep indices


def get_evidence_size(seg1_len, seg2_len):
    hard_max = seg2_len
    soft_max = seg2_len - 3
    soft_min = seg1_len
    hard_min = 1

    n_evidence_max = min(soft_max, hard_max)
    n_evidence_min = max(soft_min, hard_min)
    n_evidence_min = min(n_evidence_min, n_evidence_max)
    n_evidence_sel_size = random.randint(n_evidence_min, n_evidence_max)
    return n_evidence_sel_size


def slice_avoid_subword(tokens, st, ed):
    # If st is subword
    while 0 < st and tokens[st].startswith("##"):
        st = st - 1

    while ed < len(tokens) and tokens[ed].startswith("##"):
        ed = ed + 1

    return tokens[st: ed]


def drop_avoid_subword(tokens, st, ed):
    # If st is subword
    while st < len(tokens) and tokens[st].startswith("##"):
        st = st + 1

    ed = max(st, ed)
    while st <= ed - 1 and tokens[ed].startswith("##"):
        ed = ed - 1

    return tokens[0: st], tokens[ed:]


def select_max_range(scores, n):
    base = min(scores)
    scores_norm = [s - base for s in scores]
    max_s = -1e8
    max_i = 0
    for i in range(len(scores)):
        cur_s = sum(scores_norm[i:i+n])
        if cur_s > max_s:
            max_i = i
            max_s = cur_s
    return max_i, max_i + n


def select_min_range(scores, n):
    base = min(scores)
    scores_norm = [s+base for s in scores]
    min_s = 1e8
    min_i = 0
    for i in range(len(scores)):
        cur_s = sum(scores_norm[i:i+n])
        if cur_s < min_s:
            min_i = i
            min_s = cur_s
    return min_i, min_i + n



class EvidenceSelector:
    def get_seg(self, seg1_len: int, seg2_tokens: List[str], score: np.array) -> List[str]:
        # 1. Decide the number of tokens to be selected from seg2
        #   Similar to the length of seg1_len

        # 2. Evidence is mostly continuously selected.
        #   If short (less than half of seg2), select st:ed
        #   If long (loger than half of seg2), drop st:ed
        # 1 <= n_q_seg_tokens < ideal range < n_q_seg_tokens + 4 < n_evi_seg_tokens
        seg2_len = len(seg2_tokens)
        n_evidence_sel_size = get_evidence_size(seg1_len, seg2_len)
        if n_evidence_sel_size <= 0.5 * seg2_len:
            st, ed = select_max_range(score, n_evidence_sel_size)
            sub_tokens = slice_avoid_subword(seg2_tokens, st, ed)
        else:
            n_delete = seg2_len - n_evidence_sel_size
            st, ed = select_min_range(score, n_delete)
            head, tail = drop_avoid_subword(seg2_tokens, st, ed)
            sub_tokens = head + ["[MASK]"] + tail
        return sub_tokens


class PairWithESEncoder(PairWithESEncoderIF):
    def __init__(self, selector: EvidenceSelector, tokenizer, partitioned_encoder: PartitionedEncoderIF):
        self.partitioned_encoder: PartitionedEncoderIF = partitioned_encoder
        self.selector = selector
        self.tokenizer = tokenizer

    def select_evidence(self, e: Tuple[QueryDocES, QueryDocES]) -> \
            Tuple[BothSegPartitionedPair, BothSegPartitionedPair]:
        pos, neg = e
        assert_list_equal(pos[0].segment1.tokens, neg[0].segment1.tokens)

        def partition_pair(e) -> BothSegPartitionedPair:
            s1_part_pair, (es_score1, es_score2) = e
            s1_part_pair: Segment1PartitionedPair = s1_part_pair
            segment2_tokens: List[str] = s1_part_pair.segment2
            es_score = [es_score1, es_score2]

            parts = []
            for part_no in [0, 1]:
                seg1_len = len(s1_part_pair.segment1.get(part_no))
                seg2_tokens: List[str] = self.selector.get_seg(seg1_len, segment2_tokens, es_score[part_no])
                parts.append(seg2_tokens)

            partitioned_segment2 = MaskPartitionedSegment(*parts)
            seg_pair = BothSegPartitionedPair(s1_part_pair.segment1, partitioned_segment2, s1_part_pair.pair_data)
            return seg_pair

        return partition_pair(pos), partition_pair(neg)

    def encode_fn(self, e: Tuple[QueryDocES, QueryDocES]):
        parti_e: Tuple[BothSegPartitionedPair, BothSegPartitionedPair] = self.select_evidence(e)
        a, b = parti_e
        return self.partitioned_encoder.encode_paired(a, b)


def main():
    job_no = int(sys.argv[1])
    dataset_name = "mmp_pep_es1"

    with JobContext(f"mmp_pep_es1_{job_no}"):
        partition_len = 256
        evidence_selector = EvidenceSelector()
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        partitioned_encoder = PartitionedEncoder(tokenizer, partition_len)
        tfrecord_encoder = PairWithESEncoder(evidence_selector, tokenizer, partitioned_encoder)
        generate_train_data(job_no, dataset_name, tfrecord_encoder)


if __name__ == "__main__":
    main()
