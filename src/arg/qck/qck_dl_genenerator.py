import collections
from typing import List, Dict

from arg.qck.decl import PayloadAsTokens
from arg.qck.instance_generator.qcknc_datagen import QCKInstanceGenerator, QCKCandidateI
from data_generator.create_feature import create_int_feature
from tlm.data_gen.base import get_basic_input_feature_as_list


def encode_three_inputs(max_seq_length, tokenizer, inst: PayloadAsTokens) -> collections.OrderedDict:
    tokens_1_1: List[str] = inst.text1
    tokens_1_2: List[str] = inst.text2
    tokens_2_1: List[str] = tokens_1_2
    tokens_2_2 = inst.passage[:max_seq_length]

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

    def fill(tokens1, seg_id):
        tokens = ["[CLS]"] + tokens1 + ["[SEP]"]
        segment_ids = [seg_id] * (len(tokens1) + 2)
        tokens = tokens[:max_seq_length]
        segment_ids = segment_ids[:max_seq_length]
        return tokens, segment_ids

    tokens_A, segment_ids_A = combine(tokens_1_1, tokens_1_2)
    tokens_B, segment_ids_B = fill(tokens_2_1, 0)
    tokens_C, segment_ids_C = fill(tokens_2_2, 1)

    features = collections.OrderedDict()
    input_ids, input_mask, segment_ids = get_basic_input_feature_as_list(tokenizer, max_seq_length,
                                                                         tokens_A, segment_ids_A)
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)

    input_ids, input_mask, segment_ids = get_basic_input_feature_as_list(tokenizer, max_seq_length,
                                                                         tokens_B, segment_ids_B)
    features["input_ids1"] = create_int_feature(input_ids)
    features["input_mask1"] = create_int_feature(input_mask)
    features["segment_ids1"] = create_int_feature(segment_ids)

    input_ids, input_mask, segment_ids = get_basic_input_feature_as_list(tokenizer, max_seq_length,
                                                                         tokens_C, segment_ids_C)
    features["input_ids2"] = create_int_feature(input_ids)
    features["input_mask2"] = create_int_feature(input_mask)
    features["segment_ids2"] = create_int_feature(segment_ids)

    features['label_ids'] = create_int_feature([inst.is_correct])
    features['data_id'] = create_int_feature([inst.data_id])
    return features


class QCKDoubleLengthInstanceGenerator(QCKInstanceGenerator):
    def __init__(self,
                 candidates_dict: Dict[str, List[QCKCandidateI]],
                 is_correct_fn,
                 ):
        super(QCKDoubleLengthInstanceGenerator, self).__init__(candidates_dict, is_correct_fn)

    def encode_fn(self, inst: PayloadAsTokens) -> collections.OrderedDict:
        return encode_three_inputs(self.max_seq_length, self.tokenizer, inst)

