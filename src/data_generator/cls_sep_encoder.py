from collections import OrderedDict
from typing import List, Iterable
from typing import NamedTuple

from data_generator.create_feature import create_int_feature
from tlm.data_gen.base import get_basic_input_feature


class PairedInstance(NamedTuple):
    text1: str
    text2: str
    data_id: int
    label: int


def get_text_pair_encode_fn(max_seq_length, tokenizer):
    def encode(inst: PairedInstance) -> OrderedDict:
        tokens1: List[str] = tokenizer.tokenize(inst.text1)
        max_seg2_len = max_seq_length - 3 - len(tokens1)
        tokens2 = tokenizer.tokenize(inst.text2)[:max_seg2_len]
        tokens = ["[CLS]"] + tokens1 + ["[SEP]"] + tokens2 + ["[SEP]"]

        segment_ids = [0] * (len(tokens1) + 2) + [1] * (len(tokens2) + 1)
        tokens = tokens[:max_seq_length]
        segment_ids = segment_ids[:max_seq_length]
        features = get_basic_input_feature(tokenizer, max_seq_length, tokens, segment_ids)
        features['label_ids'] = create_int_feature([inst.label])
        features['data_id'] = create_int_feature([inst.data_id])
        return features
    return encode


def encode_save_paired_instances(data: Iterable[PairedInstance], save_path):

    pass