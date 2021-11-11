import collections
from collections import OrderedDict
from typing import List, Iterable
from typing import NamedTuple

from data_generator.create_feature import create_int_feature
from tf_util.record_writer_wrap import write_records_w_encode_fn
from tlm.data_gen.base import get_basic_input_feature, get_basic_input_feature_as_list_all_ids, \
    ordered_dict_from_input_segment_mask_ids


class ClassificationInstance(NamedTuple):
    tokens: List[str]
    seg_ids: List[int]
    label: int


class TextInstance(NamedTuple):
    text: str
    label: int
    data_id: int



class TokensAndSegmentIds(NamedTuple):
    tokens: List[str]
    seg_ids: List[int]


class InputAndSegmentIds(NamedTuple):
    input_ids: List[int]
    seg_ids: List[int]



class ClassificationInstanceWDataID(NamedTuple):
    tokens: List[str]
    seg_ids: List[int]
    label: int
    data_id: int

    @classmethod
    def make_from_tas(cls, tas: TokensAndSegmentIds, label: int, data_id: int):
        return ClassificationInstanceWDataID(tas.tokens, tas.seg_ids, label, data_id)


class PairedInstance(NamedTuple):
    tokens1: List[str]
    seg_ids1: List[int]
    tokens2: List[str]
    seg_ids2: List[int]
    data_id: int


class InstAsInputIds(NamedTuple):
    input_ids: List[int]
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


def encode_inst_as_input_ids(max_seq_length, inst: InstAsInputIds) -> OrderedDict:
    # this pads input_ids
    input_ids, input_mask, segment_ids = get_basic_input_feature_as_list_all_ids(inst.input_ids, inst.seg_ids, max_seq_length)
    feature = ordered_dict_from_input_segment_mask_ids(input_ids, input_mask, segment_ids)
    feature['label_ids'] = create_int_feature([inst.label])
    feature['data_id'] = create_int_feature([inst.data_id])
    return feature


def write_with_classification_instance_with_id(tokenizer, max_seq_length,
                                               insts: Iterable[ClassificationInstanceWDataID], out_path: str):
    def encode_fn(inst: ClassificationInstanceWDataID) -> collections.OrderedDict :
        return encode_classification_instance_w_data_id(tokenizer, max_seq_length, inst)
    write_records_w_encode_fn(out_path, encode_fn, insts)