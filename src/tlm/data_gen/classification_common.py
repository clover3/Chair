from collections import OrderedDict
from typing import List
from typing import NamedTuple

from data_generator.create_feature import create_int_feature
from tlm.data_gen.base import get_basic_input_feature


class ClassificationInstance(NamedTuple):
    tokens: List[str]
    seg_ids: List[int]
    label: int


class ClassificationInstanceWDataID(NamedTuple):
    tokens: List[str]
    seg_ids: List[int]
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

