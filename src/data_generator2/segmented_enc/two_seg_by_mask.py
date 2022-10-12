import random
from abc import ABC, abstractmethod
from typing import Tuple, List, Set

from arg.qck.encode_common import encode_single
from data_generator.tokenizer_wo_tf import get_continuation_voca_ids, FullTokenizer
from data_generator2.segmented_enc.seg_encoder_common import PairEncoderInterface, get_random_split_location
from data_generator2.segmented_enc.sent_split_by_spacy import split_spacy_tokens
from misc_lib import CountWarning, SuccessCounter
from tlm.data_gen.base import get_basic_input_feature_as_list, combine_with_sep_cls, concat_triplet_windows


def build_split_mask(tokens: List[str]) -> List[int]:
    st, ed = get_random_split_location(tokens)
    return[1 if st <= idx < ed else 0 for idx, _ in enumerate(tokens)]


def mask_encode_common(max_seq_length, tokens1, tokens2, split_mask):
    assert len(tokens2) == len(split_mask)
    max_seg2_len = max_seq_length - 3 - len(tokens1)
    tokens2 = tokens2[:max_seg2_len]
    tokens = ["[CLS]"] + tokens1 + ["[SEP]"] + tokens2 + ["[SEP]"]
    segment_ids_a = [0] * (len(tokens1) + 2)
    segment_ids_b = [1] * len(tokens2)
    segment_ids_b = [v1 + v2 for v1, v2 in zip(segment_ids_b, split_mask)]
    segment_ids = segment_ids_a + segment_ids_b + [1]
    tokens = tokens[:max_seq_length]
    segment_ids = segment_ids[:max_seq_length]
    return segment_ids, tokens


class ConcatWMask(PairEncoderInterface):
    def __init__(self, tokenizer, max_seq_len):
        self.max_seq_len = max_seq_len

        self.tokenizer = tokenizer
        self.counter_warning = CountWarning()

    def encode(self, tokens1: List[str], tokens2: List[str]) -> Tuple[List, List, List]:
        split_mask: List[int] = build_split_mask(tokens2)
        segment_ids, tokens = mask_encode_common(self.max_seq_len, tokens1, tokens2, split_mask)
        triplet = get_basic_input_feature_as_list(self.tokenizer, self.max_seq_len,
                                                  tokens, segment_ids)
        return triplet

    def encode_from_text(self, text1, text2):
        # returns input_ids, input_mask, segment_ids
        return self.encode(self.tokenizer.tokenize(text1), self.tokenizer.tokenize(text2))


class ConcatWMasInference(PairEncoderInterface):
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.counter_warning = CountWarning()
        self.max_seq_length = max_seq_length

    def encode(self, tokens1: List[str], tokens2: List[str], split_mask) -> Tuple[List, List, List]:
        segment_ids, tokens = mask_encode_common(self.max_seq_length, tokens1, tokens2, split_mask)
        triplet = get_basic_input_feature_as_list(self.tokenizer, self.max_seq_length,
                                                  tokens, segment_ids)
        return triplet

    def encode_from_text(self, text1, text2):
        # returns input_ids, input_mask, segment_ids
        return self.encode(self.tokenizer.tokenize(text1), self.tokenizer.tokenize(text2))

