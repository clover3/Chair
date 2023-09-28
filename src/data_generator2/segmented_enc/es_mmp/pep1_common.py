import pickle
from collections import OrderedDict
from typing import List, Iterable

from data_generator2.segmented_enc.es_nli.common import PHSegmentedPair
from data_generator2.segmented_enc.es_nli.path_helper import get_mmp_es0_path
from data_generator2.segmented_enc.hf_encode_helper import combine_with_sep_cls_and_pad, encode_pair
from tlm.data_gen.base import concat_tuple_windows, combine_with_sep_cls, get_basic_input_feature_as_list, \
    concat_triplet_windows
from trainer_v2.chair_logging import c_log

#
# def concat_ph_to_encode_fn(tokenizer, segment_len, e: PHSegmentedPair):
#     triplet_list = []
#     for i in [0, 1]:
#         partial_passage: List[str] = e.get_partial_prem(i)
#         partial_query: List[str] = e.get_partial_hypo(i)
#         tokens, segment_ids = combine_with_sep_cls(
#             segment_len, partial_query, partial_passage)
#         triplet = get_basic_input_feature_as_list(tokenizer, segment_len,
#                                                   tokens, segment_ids)
#         triplet_list.append(triplet)
#     triplet = concat_triplet_windows(triplet_list, segment_len)
#     return triplet


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

