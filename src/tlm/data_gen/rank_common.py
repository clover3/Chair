import collections
from collections import OrderedDict
from typing import NamedTuple, List

from data_generator.create_feature import create_int_feature
from tlm.data_gen.base import get_basic_input_feature_as_list


class QueryDocInstance(NamedTuple):
    query_tokens: List[str]
    doc_tokens: List[str]
    label: int
    data_id: int


class QueryDocPairInstance(NamedTuple):
    query_tokens: List[str]
    doc_tokens1: List[str]
    doc_tokens2: List[str]
    data_id: int


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


def encode_query_doc_pair_instance(tokenizer, inst: QueryDocPairInstance) -> OrderedDict:
    def token_to_feature(tokens):
        return create_int_feature(tokenizer.convert_tokens_to_ids(tokens))

    feature = collections.OrderedDict()
    feature['query'] = token_to_feature(inst.query_tokens)
    feature['doc1'] = token_to_feature(inst.doc_tokens1)
    feature['doc2'] = token_to_feature(inst.doc_tokens2)
    feature['data_id'] = create_int_feature([inst.data_id])
    return feature

