import collections
from collections import OrderedDict
from typing import List
from typing import NamedTuple

from data_generator.create_feature import create_int_feature
from tlm.data_gen.base import get_basic_input_feature, get_basic_input_feature_as_list


class ClassificationInstance(NamedTuple):
    tokens: List[str]
    seg_ids: List[int]
    label: int


class TokensAndSegmentIds(NamedTuple):
    tokens: List[str]
    seg_ids: List[int]


class ClassificationInstanceWDataID(NamedTuple):
    tokens: List[str]
    seg_ids: List[int]
    label: int
    data_id: int

    @classmethod
    def make_from_tas(cls, tas: TokensAndSegmentIds, label: int, data_id: int):
        return ClassificationInstanceWDataID(tas.tokens, tas.seg_ids, label, data_id)


class QueryDocInstance(NamedTuple):
    query_tokens: List[str]
    doc_tokens: List[str]
    label: int
    data_id: int


def encode_classification_instance(tokenizer, max_seq_length, inst: ClassificationInstance) -> OrderedDict:
    feature: OrderedDict = get_basic_input_feature(tokenizer, max_seq_length, inst.tokens, inst.seg_ids)
    feature['label_ids'] = create_int_feature([inst.label])
    return feature


def encode_classification_instance_w_data_id(tokenizer, max_seq_length, inst: ClassificationInstanceWDataID) -> OrderedDict:
    feature: OrderedDict = get_basic_input_feature(tokenizer, max_seq_length, inst.tokens, inst.seg_ids)
    feature['label_ids'] = create_int_feature([inst.label])
    feature['data_id'] = create_int_feature([inst.data_id])
    return feature


def encode_query_doc_instance(tokenizer, doc_token_length, inst: QueryDocInstance) -> OrderedDict:
    doc_segment_ids = [1] * len(inst.doc_tokens)
    doc_input_ids, doc_input_mask, doc_segment_ids \
        = get_basic_input_feature_as_list(tokenizer, doc_token_length, inst.doc_tokens, doc_segment_ids)

    feature = collections.OrderedDict()
    feature['query'] = create_int_feature(tokenizer.convert_tokens_to_ids(inst.query_tokens))
    feature['doc'] = create_int_feature(doc_input_ids)
    feature['doc_mask'] = create_int_feature(doc_input_mask)
    feature['label_ids'] = create_int_feature([inst.label])
    feature['data_id'] = create_int_feature([inst.data_id])
    return feature
