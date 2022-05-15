import collections
from collections import OrderedDict
from typing import List

from arg.qck.decl import PayloadAsTokens, PayloadAsIds
from data_generator.create_feature import create_int_feature
from tlm.data_gen.base import get_basic_input_feature_as_list_all_ids, get_basic_input_feature_as_list
from tlm.data_gen.pairwise_common import combine_features_B


def encode_two_inputs(max_seq_length, tokenizer, inst: PayloadAsTokens) -> OrderedDict:
    tokens_1_1: List[str] = inst.text1
    tokens_1_2: List[str] = inst.text2
    tokens_2_1: List[str] = tokens_1_2

    max_seg2_len = max_seq_length - 3 - len(tokens_2_1)

    tokens_2_2 = inst.passage[:max_seg2_len]

    def combine(tokens1, tokens2):
        effective_length = max_seq_length - 3
        if len(tokens1) + len(tokens2) > effective_length:
            half = int(effective_length/2 + 1)
            tokens1 = tokens1[:half]
            remain = effective_length - len(tokens1)
            tokens2 = tokens2[:remain]
        tokens = ["[CLS]"] + tokens1 + ["[SEP]"] + tokens2 + ["[SEP]"]
        segment_ids = [0] * (len(tokens1) + 2) \
                      + [1] * (len(tokens2) + 1)
        tokens = tokens[:max_seq_length]
        segment_ids = segment_ids[:max_seq_length]
        return tokens, segment_ids

    tokens_A, segment_ids_A = combine(tokens_1_1, tokens_1_2)
    tokens_B, segment_ids_B = combine(tokens_2_1, tokens_2_2)

    features = combine_features_B(tokens_A, segment_ids_A, tokens_B, segment_ids_B, tokenizer, max_seq_length)
    features['label_ids'] = create_int_feature([inst.is_correct])
    features['data_id'] = create_int_feature([inst.data_id])
    return features


def encode_two_input_ids(max_seq_length, tokenizer, inst: PayloadAsIds) -> OrderedDict:
    tokens_1_1: List[int] = inst.text1
    tokens_1_2: List[int] = inst.text2
    tokens_2_1: List[int] = tokens_1_2

    cls_id = tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
    sep_id = tokenizer.convert_tokens_to_ids(["[SEP]"])[0]
    max_seg2_len = max_seq_length - 3 - len(tokens_2_1)

    tokens_2_2 = inst.passage[:max_seg2_len]

    def combine(tokens1, tokens2):
        effective_length = max_seq_length - 3
        if len(tokens1) + len(tokens2) > effective_length:
            half = int(effective_length/2 + 1)
            tokens1 = tokens1[:half]
            remain = effective_length - len(tokens1)
            tokens2 = tokens2[:remain]
        input_ids = [cls_id] + tokens1 + [sep_id] + tokens2 + [sep_id]
        segment_ids = [0] * (len(tokens1) + 2) + [1] * (len(tokens2) + 1)
        input_ids = input_ids[:max_seq_length]
        segment_ids = segment_ids[:max_seq_length]
        return input_ids, segment_ids

    input_ids_A, segment_ids_A = combine(tokens_1_1, tokens_1_2)
    input_ids_B, segment_ids_B = combine(tokens_2_1, tokens_2_2)

    input_ids, input_mask, segment_ids = get_basic_input_feature_as_list_all_ids(input_ids_A, segment_ids_A,
                                                                                 max_seq_length)
    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)

    input_ids, input_mask, segment_ids = get_basic_input_feature_as_list_all_ids(input_ids_B, segment_ids_B,
                                                                                 max_seq_length)
    features["input_ids2"] = create_int_feature(input_ids)
    features["input_mask2"] = create_int_feature(input_mask)
    features["segment_ids2"] = create_int_feature(segment_ids)

    features['label_ids'] = create_int_feature([inst.is_correct])
    features['data_id'] = create_int_feature([inst.data_id])
    return features


def encode_single(tokenizer, tokens, max_seq_length):
    effective_length = max_seq_length - 2
    tokens = tokens[:effective_length]
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    segment_ids = [0] * len(tokens)
    tokens = tokens[:max_seq_length]
    segment_ids = segment_ids[:max_seq_length]
    input_ids, input_mask, segment_ids = get_basic_input_feature_as_list(tokenizer, max_seq_length,
                                                                         tokens, segment_ids)

    return input_ids, input_mask, segment_ids


def encode_three_inputs(max_seq_length_list: List[int], tokenizer, inst: PayloadAsTokens) -> OrderedDict:
    tokens1: List[str] = inst.text1
    tokens2: List[str] = inst.text2
    tokens3: List[str] = inst.passage

    tokens_list = [tokens1, tokens2, tokens3]
    features = collections.OrderedDict()
    for i in range(3):
        input_ids, input_mask, segment_ids = encode_single(tokenizer, tokens_list[i], max_seq_length_list[i])
        features["input_ids{}".format(i)] = input_ids
        features["input_mask{}".format(i)] = input_mask
        features["segment_ids{}".format(i)] = segment_ids

    features['label_ids'] = create_int_feature([inst.is_correct])
    features['data_id'] = create_int_feature([inst.data_id])
    return features

