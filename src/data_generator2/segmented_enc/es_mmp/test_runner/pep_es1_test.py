import numpy as np
from transformers import AutoTokenizer

from data_generator.tokenizer_wo_tf import pretty_tokens
from data_generator2.segmented_enc.es_common.es_two_seg_common import BothSegPartitionedPair, Segment1PartitionedPair, \
    MaskPartitionedSegment
from data_generator2.segmented_enc.es_common.partitioned_encoder import PartitionedEncoder
from data_generator2.segmented_enc.es_mmp.pep_es_common import iter_es_data_pos_neg_pair, QueryDocES, \
    PairWithESEncoderIF
from data_generator2.segmented_enc.es_mmp.runner.pep_es1_gen import EvidenceSelector, PairWithESEncoder, \
    get_evidence_size, select_max_range, slice_avoid_subword, select_min_range, drop_avoid_subword
from typing import List, Iterable, Callable, Dict, Tuple, Set

from list_lib import assert_list_equal
from misc_lib import two_digit_float
from tab_print import tab_print, print_table


def print_lists_horizontally(list1, list2, column_length=10):
    # Calculate the max width of the strings to align properly
    cursor = 0
    while cursor < len(list1):
        table = [list1[cursor:cursor + column_length],
                 list2[cursor:cursor + column_length]]
        print_table(table)
        cursor += column_length


def generate_train_data(
        job_no: int, tfrecord_encoder: PairWithESEncoder):
    # Mocking the iteration over dataset and print the tokens selected for evidence.
    pos_neg_itr: Iterable[Tuple[QueryDocES, QueryDocES]] = iter_es_data_pos_neg_pair(job_no)
    def get_seg(seg1_len: int, seg2_tokens: List[str], score: np.array) -> List[str]:
        # 1. Decide the number of tokens to be selected from seg2
        #   Similar to the length of seg1_len

        # 2. Evidence is mostly continuously selected.
        #   If short (less than half of seg2), select st:ed
        #   If long (loger than half of seg2), drop st:ed
        # 1 <= n_q_seg_tokens < ideal range < n_q_seg_tokens + 4 < n_evi_seg_tokens
        seg2_len = len(seg2_tokens)
        n_evidence_sel_size = get_evidence_size(seg1_len, seg2_len)
        print("get_seg ENTRY")
        print(f"seg1/seg2/n_evidence = {seg1_len}/{seg2_len}/{n_evidence_sel_size}")
        score_str_l = list(map(two_digit_float, score))
        print_lists_horizontally(seg2_tokens, score_str_l)
        if n_evidence_sel_size <= 0.5 * seg2_len:
            st, ed = select_max_range(score, n_evidence_sel_size)
            sub_tokens = slice_avoid_subword(seg2_tokens, st, ed)
        else:
            n_delete = seg2_len - n_evidence_sel_size
            st, ed = select_min_range(score, n_delete)
            head, tail = drop_avoid_subword(seg2_tokens, st, ed)
            sub_tokens = head + ["[MASK]"] + tail
        return sub_tokens

    cnt = 0
    for pos_neg_pair in pos_neg_itr:
        cnt += 1
        if cnt > 10:
            break
        # We'll process each pair and print the result instead of writing to a file.
        pos, neg = pos_neg_pair
        print()
        print("< Paired data > ")
        assert_list_equal(pos[0].segment1.tokens, neg[0].segment1.tokens)

        def partition_pair(e) -> BothSegPartitionedPair:
            s1_part_pair, (es_score1, es_score2) = e
            s1_part_pair: Segment1PartitionedPair = s1_part_pair
            segment2_tokens: List[str] = s1_part_pair.segment2
            es_score = [es_score1, es_score2]

            parts = []
            for part_no in [0, 1]:
                print("Part ", part_no)
                seg1_len = len(s1_part_pair.segment1.get(part_no))
                print("seg1:", s1_part_pair.segment1.get(part_no))
                seg2_tokens: List[str] = get_seg(seg1_len, segment2_tokens, es_score[part_no])
                print("selected:", pretty_tokens(seg2_tokens, True))
                parts.append(seg2_tokens)

            partitioned_segment2 = MaskPartitionedSegment(*parts)
            seg_pair = BothSegPartitionedPair(s1_part_pair.segment1, partitioned_segment2, s1_part_pair.pair_data)
            return seg_pair


        ret = partition_pair(pos)
        print("Selected tokens for pos example:\n", ret)
        ret = partition_pair(neg)
        print("Selected tokens for neg example:\n", ret)


def main():
    # job_no and dataset_name are now hardcoded for demonstration purposes.
    job_no = 1
    partition_len = 256
    evidence_selector = EvidenceSelector()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    partitioned_encoder = PartitionedEncoder(tokenizer, partition_len)
    tfrecord_encoder = PairWithESEncoder(evidence_selector, tokenizer, partitioned_encoder)
    generate_train_data(job_no, tfrecord_encoder)


if __name__ == "__main__":
    main()