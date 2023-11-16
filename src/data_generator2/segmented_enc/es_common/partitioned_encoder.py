from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List
from typing import List, Iterable, Callable, Dict, Tuple, Set

import numpy as np

from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.es_common.es_two_seg_common import PartitionedSegment, RangePartitionedSegment, \
    IndicesPartitionedSegment, BothSegPartitionedPair, PairData, Segment1PartitionedPair, EvidencePair2, \
    MaskPartitionedSegment, compress_mask
from data_generator2.segmented_enc.hf_encode_helper import combine_with_sep_cls_and_pad, encode_pair, \
    encode_seg_pair_paired
from data_generator2.segmented_enc.seg_encoder_common import get_random_split_location
from tlm.data_gen.base import concat_tuple_windows
from trainer_v2.chair_logging import c_log
from trainer_v2.evidence_selector.evidence_candidates import get_st_ed


def segment_formatter(e: PartitionedSegment, target_part_no):
    if isinstance(e, RangePartitionedSegment):
        head, tail = e.get_first()
        if target_part_no == 0:
            return head + ["[MASK]"] + tail
        elif target_part_no == 1:
            head_mask = ["[MASK]"] if head else []
            tail_mask = ["[MASK]"] if tail else []
            return head_mask + e.get_second() + tail_mask
        else:
            assert False
    elif isinstance(e, IndicesPartitionedSegment):
        return e.get_partition_seg(target_part_no)
    elif isinstance(e, MaskPartitionedSegment):
        # Supposed to have mask at boundary of d tokens
        return e.get_partition_seg(target_part_no)
    else:
        assert False


def segment_formatter2(e: PartitionedSegment, target_part_no):
    if isinstance(e, RangePartitionedSegment):
        head, tail = e.get_first()
        if target_part_no == 0:
            return head + ["[MASK]"] + tail
        elif target_part_no == 1:
            return e.get_second()
        else:
            assert False
    elif isinstance(e, IndicesPartitionedSegment):
        return e.get_partition_seg(target_part_no)
    elif isinstance(e, MaskPartitionedSegment):
        # Supposed to have mask at boundary of d tokens
        return e.get_partition_seg(target_part_no)
    else:
        assert False


class PartitionedEncoderIF(ABC):
    @abstractmethod
    def encode(self, e: BothSegPartitionedPair) -> OrderedDict:
        pass

    @abstractmethod
    def encode_paired(
            self, pos: BothSegPartitionedPair, neg: BothSegPartitionedPair) -> OrderedDict:
        pass


class PartitionedEncoder(PartitionedEncoderIF):
    def __init__(self, tokenizer, segment_len: int):
        self.tokenizer = tokenizer
        self.partition_len = segment_len

    def encode_to_ids(self, e: BothSegPartitionedPair):
        partition_len = self.partition_len
        tuple_list = []
        for part_no in [0, 1]:
            partial_seg1: List[str] = segment_formatter(e.segment1, part_no)
            partial_seg2: List[str] = segment_formatter(e.segment2, part_no)
            input_ids, segment_ids = combine_with_sep_cls_and_pad(
                self.tokenizer, partial_seg1, partial_seg2, partition_len)
            tuple_list.append((input_ids, segment_ids))
        return concat_tuple_windows(tuple_list, partition_len)

    def encode(self, e: BothSegPartitionedPair) -> OrderedDict:
        input_ids_segment_ids = self.encode_to_ids(e)
        return encode_pair(input_ids_segment_ids, int(e.pair_data.label))

    def encode_paired(
            self, pos: BothSegPartitionedPair, neg: BothSegPartitionedPair) -> OrderedDict:
        return encode_seg_pair_paired(self.encode_to_ids(pos), self.encode_to_ids(neg))


def segment_formatter_w_compress(segment, part_no):
    seg: List[str] = segment_formatter(segment, part_no)
    return compress_mask(seg)


class PartitionedEncoderCompressMask(PartitionedEncoderIF):
    def __init__(self, tokenizer, segment_len: int):
        self.tokenizer = tokenizer
        self.partition_len = segment_len

    def encode_to_ids(self, e: BothSegPartitionedPair):
        partition_len = self.partition_len
        tuple_list = []
        for part_no in [0, 1]:
            partial_seg1: List[str] = segment_formatter_w_compress(e.segment1, part_no)
            partial_seg2: List[str] = segment_formatter_w_compress(e.segment2, part_no)
            input_ids, segment_ids = combine_with_sep_cls_and_pad(
                self.tokenizer, partial_seg1, partial_seg2, partition_len)
            tuple_list.append((input_ids, segment_ids))
        return concat_tuple_windows(tuple_list, partition_len)

    def encode(self, e: BothSegPartitionedPair) -> OrderedDict:
        input_ids_segment_ids = self.encode_to_ids(e)
        return encode_pair(input_ids_segment_ids, int(e.pair_data.label))

    def encode_paired(
            self, pos: BothSegPartitionedPair, neg: BothSegPartitionedPair) -> OrderedDict:
        return encode_seg_pair_paired(self.encode_to_ids(pos), self.encode_to_ids(neg))


def get_both_seg_partitioned_to_input_ids(tokenizer, parition_len: int):
    def encode(e: BothSegPartitionedPair):
        tuple_list = []
        for part_no in [0, 1]:
            partial_seg1: List[str] = segment_formatter(e.segment1, part_no)
            partial_seg2: List[str] = segment_formatter(e.segment2, part_no)
            input_ids, segment_ids = combine_with_sep_cls_and_pad(
                tokenizer, partial_seg1, partial_seg2, parition_len)
            tuple_list.append((input_ids, segment_ids))
        return concat_tuple_windows(tuple_list, parition_len)
    return encode


def get_both_seg_partitioned_to_input_ids2(tokenizer, partition_len: int) \
    -> Callable[[BothSegPartitionedPair], Iterable[tuple[list, list]]]:
    def encode(e: BothSegPartitionedPair):
        for part_no in [0, 1]:
            partial_seg1: List[str] = segment_formatter(e.segment1, part_no)
            partial_seg2: List[str] = segment_formatter(e.segment2, part_no)
            input_ids, segment_ids = combine_with_sep_cls_and_pad(
                tokenizer, partial_seg1, partial_seg2, partition_len)
            yield input_ids, segment_ids
    return encode


def apply_segmentation_to_seg1(tokenizer, item: PairData) -> Segment1PartitionedPair:
    segment1_tokens = tokenizer.tokenize(item.segment1)
    segment2_tokens: List[str] = tokenizer.tokenize(item.segment2)
    st, ed = get_random_split_location(segment1_tokens)
    segment1 = RangePartitionedSegment(segment1_tokens, st, ed)
    return Segment1PartitionedPair(segment1, segment2_tokens, item)


class BothSegPartitionedPairParser:
    def __init__(self, segment_len):
        self.segment_len = segment_len
        self.tokenizer = get_tokenizer()
        self.mask_id = self.tokenizer.wordpiece_tokenizer.vocab["[MASK]"]

    def parse(self, input_ids, segment_ids) -> EvidencePair2:
        input_ids = input_ids.numpy().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        segment_ids = segment_ids.numpy().tolist()

        tokens1: List[str] = tokens[:self.segment_len]
        segment_ids1 = segment_ids[:self.segment_len]

        tokens2 = tokens[self.segment_len:]
        segment_ids2 = segment_ids[self.segment_len:]

        def split(input_ids, segment_ids):
            seg2_start, seg2_end = get_st_ed(segment_ids)
            seg1 = input_ids[1: seg2_start-1]
            seg2 = input_ids[seg2_start: seg2_end-1]
            return seg1, seg2

        seg1_part1, seg2_part1 = split(tokens1, segment_ids1)
        seg1_part2, seg2_part2 = split(tokens2, segment_ids2)

        range_part_segs = [seg1_part1, seg1_part2]
        indice_part_segs = [seg2_part1, seg2_part2]
        if not len(indice_part_segs[0]) == len(indice_part_segs[1]):
            len1 = 3 + len(indice_part_segs[0]) + len(range_part_segs[0])
            len2 = 3 + len(indice_part_segs[1]) + len(range_part_segs[1])
            if len1 == self.segment_len or len2 == self.segment_len:
                pass
            else:
                c_log.warning("%d differs %d. ", len(indice_part_segs[0]), len(indice_part_segs[1]))
                c_log.warning("Length for other part : %d, %d", len(range_part_segs[0]), len(range_part_segs[1]))

        indices_part_merged: List[str] = self.merge_indices_part(indice_part_segs)
        mask_segment = MaskPartitionedSegment(*range_part_segs)

        def get_del_indices(p_i):
            indices = []
            for i, t in enumerate(p_i):
                if t == "[MASK]":
                    indices.append(i)
            return indices

        indices_segment = IndicesPartitionedSegment(
            indices_part_merged,
            get_del_indices(indice_part_segs[0]),
            get_del_indices(indice_part_segs[1]),
        )

        return EvidencePair2(mask_segment, indices_segment)

    def merge_indices_part(self, indice_part_segs: List[str]) -> List[str]:
        indices_part_merged = []
        for i, t in enumerate(indice_part_segs[0]):
            if t == "[MASK]":
                indices_part_merged.append(indice_part_segs[1][i])
            else:
                indices_part_merged.append(indice_part_segs[0][i])
        return indices_part_merged


# cur len = 3,
# cont_seg_len = 49
def build_get_num_delete_fn(del_rate: float):
    def get_num_delete(cur_part_len, other_part_len, cont_seg_len):
        # Assume each token of current part require one token in continuous one
        # For example,
        # cont_seg_len = 40 (document
        # cur_part_len = 5 (query part 1)
        # Then we need at least 5 and we can delete at most 35
        # if del_rate is 0.5, normal_mean = std_dev = 17.5

        hard_min_requirement = max(cur_part_len// 2, 1)
        hard_max_del = cont_seg_len - hard_min_requirement
        soft_min_requirement = cur_part_len
        reasonable_max_del = cont_seg_len - soft_min_requirement
        reasonable_max_del = max(reasonable_max_del, 1)

        normal_mean = reasonable_max_del * del_rate
        std_dev = normal_mean
        num_del = int(np.random.normal(normal_mean, std_dev))

        # Cannot delete more than cont_seg_len
        num_del = min(hard_max_del, num_del)
        num_del = max(0, num_del)
        return num_del
    return get_num_delete