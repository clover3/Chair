import collections
from collections import OrderedDict
from typing import NamedTuple, List

from data_generator.create_feature import create_int_feature, create_float_feature
from tlm.data_gen.classification_common import TokensAndSegmentIds, InputAndSegmentIds


class QueryEntityDocPairInstance(NamedTuple):
    query_tokens: List[str]
    entity: List[str]
    doc_tokens1: List[str]
    doc_tokens2: List[str]
    data_id: int


class QDE(NamedTuple):
    q_e: TokensAndSegmentIds
    d_e: TokensAndSegmentIds


class QDE_as_Ids(NamedTuple):
    q_e: InputAndSegmentIds
    d_e: InputAndSegmentIds
    score: float
    data_id: int


class QTypeDE_as_Ids(NamedTuple):
    qtype_id: int
    d_e: InputAndSegmentIds
    score: float
    data_id: int


class QDEPaired(NamedTuple):
    qde1: QDE
    qde2: QDE
    data_id: int


def encode_query_entity_doc_pair_instance(tokenizer, inst: QueryEntityDocPairInstance) -> OrderedDict:
    def token_to_feature(tokens):
        return create_int_feature(tokenizer.convert_tokens_to_ids(tokens))

    feature = collections.OrderedDict()
    feature['query'] = token_to_feature(inst.query_tokens)
    feature['entity'] = token_to_feature(inst.entity)
    feature['doc1'] = token_to_feature(inst.doc_tokens1)
    feature['doc2'] = token_to_feature(inst.doc_tokens2)
    feature['data_id'] = create_int_feature([inst.data_id])
    return feature



def encode_concatenated_query_entity_doc_pair_instance(tokenizer,
                                                       max_seq_length,
                                                       q_max_seq_length,
                                                       inst: QDEPaired) -> OrderedDict:
    def pad_to_length(seq, pad_len):
        seq = seq[:pad_len]
        n_pad = pad_len - len(seq)
        return seq + [0] * n_pad

    def convert_pad_assign(qd: TokensAndSegmentIds, pad_len):
        input_ids = tokenizer.convert_tokens_to_ids(qd.tokens)
        input_ids = create_int_feature(pad_to_length(input_ids, pad_len))
        segment_ids = create_int_feature(pad_to_length(qd.seg_ids, pad_len))
        return input_ids, segment_ids

    def qde_to_od(qde: QDE):
        feature = collections.OrderedDict()
        input_ids, segment_ids = convert_pad_assign(qde.q_e, q_max_seq_length)
        feature['q_e_input_ids'] = input_ids
        feature['q_e_segment_ids'] = segment_ids
        input_ids, segment_ids = convert_pad_assign(qde.d_e, max_seq_length)
        feature['d_e_input_ids'] = input_ids
        feature['d_e_segment_ids'] = segment_ids
        return feature

    feature = collections.OrderedDict()
    todo = [
        (inst.qde1, "1"),
        (inst.qde2, "2")
    ]

    for qde, post_fix in todo:
        od = qde_to_od(qde)
        for key in od.keys():
            feature[key + post_fix] = od[key]

    feature['data_id'] = create_int_feature([inst.data_id])
    return feature


def encode_qde_ids_instance(
                           max_seq_length,
                           q_max_seq_length,
                           inst: QDE_as_Ids) -> OrderedDict:
    def pad_to_length(seq, pad_len):
        seq = seq[:pad_len]
        n_pad = pad_len - len(seq)
        return seq + [0] * n_pad

    def convert_pad_assign(qd: InputAndSegmentIds, pad_len):
        input_ids = create_int_feature(pad_to_length(qd.input_ids, pad_len))
        segment_ids = create_int_feature(pad_to_length(qd.seg_ids, pad_len))
        return input_ids, segment_ids

    feature = collections.OrderedDict()
    input_ids, segment_ids = convert_pad_assign(inst.q_e, q_max_seq_length)
    feature['q_e_input_ids'] = input_ids
    feature['q_e_segment_ids'] = segment_ids
    input_ids, segment_ids = convert_pad_assign(inst.d_e, max_seq_length)
    feature['d_e_input_ids'] = input_ids
    feature['d_e_segment_ids'] = segment_ids
    feature['label_ids'] = create_float_feature([inst.score])
    feature['data_id'] = create_int_feature([inst.data_id])
    return feature


def encode_qtype_de_ids_instance(
        max_seq_length,
        inst: QTypeDE_as_Ids) -> OrderedDict:
    def pad_to_length(seq, pad_len):
        seq = seq[:pad_len]
        n_pad = pad_len - len(seq)
        return seq + [0] * n_pad

    def convert_pad_assign(qd: InputAndSegmentIds, pad_len):
        input_ids = create_int_feature(pad_to_length(qd.input_ids, pad_len))
        segment_ids = create_int_feature(pad_to_length(qd.seg_ids, pad_len))
        return input_ids, segment_ids

    feature = collections.OrderedDict()
    feature['qtype_id'] = create_int_feature([inst.qtype_id])
    input_ids, segment_ids = convert_pad_assign(inst.d_e, max_seq_length)
    feature['d_e_input_ids'] = input_ids
    feature['d_e_segment_ids'] = segment_ids
    feature['label_ids'] = create_float_feature([inst.score])
    feature['data_id'] = create_int_feature([inst.data_id])
    return feature