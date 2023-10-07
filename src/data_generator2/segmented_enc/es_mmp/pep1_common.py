import pickle
from collections import OrderedDict
from typing import List

from data_generator2.segmented_enc.es_nli.common import PHSegmentedPair
from data_generator2.segmented_enc.es_nli.path_helper import get_mmp_es0_path
from data_generator2.segmented_enc.es_common.es_two_seg_common import BothSegPartitionedPair, RangePartitionedSegment, \
    PartitionedSegment, IndicesPartitionedSegment
from data_generator2.segmented_enc.hf_encode_helper import combine_with_sep_cls_and_pad, encode_pair
from tlm.data_gen.base import concat_tuple_windows
from trainer_v2.chair_logging import c_log


def get_ph_segment_pair_encode_fn(tokenizer, segment_len: int):
    def encode_fn(e: PHSegmentedPair) -> OrderedDict:
        tuple_list = []
        for i in [0, 1]:
            partial_passage: List[str] = e.get_partial_prem(i)
            partial_query: List[str] = e.get_partial_hypo(i)
            input_ids, segment_ids = combine_with_sep_cls_and_pad(
                tokenizer, partial_query, partial_passage, segment_len)
            tuple_list.append((input_ids, segment_ids))
        triplet = concat_tuple_windows(tuple_list, segment_len)
        return encode_pair(triplet, int(e.nli_pair.label))
    return encode_fn


def load_ph_segmented_pair(partition_no) -> List[PHSegmentedPair]:
    source_path = get_mmp_es0_path(partition_no)
    c_log.info("Loading pickle from %s", source_path)
    payload: List[PHSegmentedPair] = pickle.load(open(source_path, "rb"))
    c_log.debug("Done")
    return payload


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
    else:
        assert False


def get_both_seg_partitioned_pair_encode_fn(tokenizer, segment_len: int):
    def encode_fn(e: BothSegPartitionedPair) -> OrderedDict:
        tuple_list = []
        for part_no in [0, 1]:
            partial_seg1: List[str] = segment_formatter(e.segment1, part_no)
            partial_seg2: List[str] = segment_formatter(e.segment2, part_no)
            input_ids, segment_ids = combine_with_sep_cls_and_pad(
                tokenizer, partial_seg1, partial_seg2, segment_len)
            tuple_list.append((input_ids, segment_ids))
        triplet = concat_tuple_windows(tuple_list, segment_len)
        return encode_pair(triplet, int(e.pair_data.label))
    return encode_fn
