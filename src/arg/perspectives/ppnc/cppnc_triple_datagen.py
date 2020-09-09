import collections
from collections import OrderedDict
from typing import List

from arg.perspectives.ppnc.cppnc_datagen import convert_sub_token, Payload
from arg.perspectives.ppnc.ppnc_decl import PayloadAsTokens
from data_generator.create_feature import create_int_feature
from data_generator.tokenizer_wo_tf import get_tokenizer
from tf_util.record_writer_wrap import write_records_w_encode_fn
from tlm.data_gen.base import get_basic_input_feature_as_list


def encode_inner(max_seq_length, tokenizer, inst: PayloadAsTokens) -> OrderedDict:
    tokens_1: List[str] = inst.text1
    tokens_2: List[str] = inst.text2
    tokens_3: List[str] = inst.passage

    def combine(tokens1, tokens2):
        max_seg2_len = max_seq_length - 3 - len(tokens1)
        tokens2 = tokens2[:max_seg2_len]
        tokens = ["[CLS]"] + tokens1 + ["[SEP]"] + tokens2 + ["[SEP]"]
        segment_ids = [0] * (len(tokens1) + 2) \
                      + [1] * (len(tokens2) + 1)
        tokens = tokens[:max_seq_length]
        segment_ids = segment_ids[:max_seq_length]
        return tokens, segment_ids

    features = collections.OrderedDict()
    for tokens_a, tokens_b, postfix in [(tokens_1, tokens_2, ""),
                               (tokens_2, tokens_3, "2"),
                               (tokens_1, tokens_3, "3")]:
        tokens, segment_ids = combine(tokens_a, tokens_b)
        input_ids, input_mask, segment_ids = get_basic_input_feature_as_list(tokenizer, max_seq_length,
                                                                             tokens, segment_ids)

        features["input_ids" + postfix] = create_int_feature(input_ids)
        features["input_mask" + postfix] = create_int_feature(input_mask)
        features["segment_ids" + postfix] = create_int_feature(segment_ids)

    features['label_ids'] = create_int_feature([inst.is_correct])
    features['data_id'] = create_int_feature([inst.data_id])
    return features


def write_records(records: List[Payload],
                  max_seq_length,
                  output_path):
    tokenizer = get_tokenizer()

    def encode(inst: Payload) -> OrderedDict:
        inst_2 = convert_sub_token(tokenizer, inst)
        return encode_inner(max_seq_length, tokenizer, inst_2)

    write_records_w_encode_fn(output_path, encode, records)
