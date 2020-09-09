from collections import OrderedDict
from typing import List

from arg.perspectives.ppnc.ppnc_decl import PayloadAsTokens
from data_generator.create_feature import create_int_feature
from tlm.data_gen.pairwise_common import combine_features_B


def encode_two_inputs(max_seq_length, tokenizer, inst: PayloadAsTokens) -> OrderedDict:
    tokens_1_1: List[str] = inst.text1
    tokens_1_2: List[str] = inst.text2
    tokens_2_1: List[str] = tokens_1_2

    max_seg2_len = max_seq_length - 3 - len(tokens_2_1)

    tokens_2_2 = inst.passage[:max_seg2_len]

    def combine(tokens1, tokens2):
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