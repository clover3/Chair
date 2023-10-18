import pickle
from collections import OrderedDict

from data_generator2.segmented_enc.es_nli.common import PHSegmentedPair
from data_generator2.segmented_enc.es_nli.path_helper import get_mmp_es0_path
from data_generator2.segmented_enc.hf_encode_helper import combine_with_sep_cls_and_pad, encode_pair
from dataset_specific.mnli.mnli_reader import NLIPairData
from tlm.data_gen.base import concat_tuple_windows
from trainer_v2.chair_logging import c_log
from typing import List, Callable, Tuple
from warnings import warn

from trainer_v2.per_project.cip.cip_common import get_random_split_location


def get_ph_segment_pair_encode_fn(tokenizer, segment_len: int) -> Callable[[PHSegmentedPair], OrderedDict]:
    new_target = "data_generator2.segmented_enc.es_common.partitioned_encoder.get_both_seg_partitioned_pair_encode_fn"
    warn('This is deprecated. use ' + new_target , DeprecationWarning, stacklevel=2)

    def encode_fn(e: PHSegmentedPair) -> OrderedDict:
        triplet = tokenize_encode_ph_segmented_pair(e)
        return encode_pair(triplet, int(e.nli_pair.label))

    def tokenize_encode_ph_segmented_pair(e):
        tuple_list = []
        for i in [0, 1]:
            partial_passage: List[str] = e.get_partial_prem(i)
            partial_query: List[str] = e.get_partial_hypo(i)
            input_ids, segment_ids = combine_with_sep_cls_and_pad(
                tokenizer, partial_query, partial_passage, segment_len)
            tuple_list.append((input_ids, segment_ids))
        triplet = concat_tuple_windows(tuple_list, segment_len)
        return triplet

    return encode_fn


def ph_segment_to_input_ids(tokenizer, segment_len: int) -> Callable[[PHSegmentedPair], Tuple]:
    def func(e) -> Tuple:
        tuple_list = []
        for i in [0, 1]:
            partial_passage: List[str] = e.get_partial_prem(i)
            partial_query: List[str] = e.get_partial_hypo(i)
            input_ids, segment_ids = combine_with_sep_cls_and_pad(
                tokenizer, partial_query, partial_passage, segment_len)
            tuple_list.append((input_ids, segment_ids))
        input_ids, segment_ids = concat_tuple_windows(tuple_list, segment_len)
        return input_ids, segment_ids
    return func


def load_ph_segmented_pair(partition_no) -> List[PHSegmentedPair]:
    source_path = get_mmp_es0_path(partition_no)
    c_log.info("Loading pickle from %s", source_path)
    payload: List[PHSegmentedPair] = pickle.load(open(source_path, "rb"))
    c_log.debug("Done")
    return payload


def partition_query_ph_based(
        tokenizer, qd_pair: Tuple[str, str]) -> PHSegmentedPair:
    query, document = qd_pair
    q_tokens = tokenizer.tokenize(query)
    d_tokens = tokenizer.tokenize(document)
    h_st, h_ed = get_random_split_location(q_tokens)
    nli_pair = NLIPairData(document, query, "0", "0")
    ph_seg_pair = PHSegmentedPair(d_tokens, q_tokens, h_st, h_ed, [], [], nli_pair)
    return ph_seg_pair