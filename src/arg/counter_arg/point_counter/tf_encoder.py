from collections import OrderedDict
from typing import Tuple

from data_generator.create_feature import create_int_feature
from data_generator.tokenizer_wo_tf import get_tokenizer
from tlm.data_gen.base import get_basic_input_feature
from tlm.data_gen.classification_common import TextInstance


def get_encode_fn(max_seq_length):
    tokenizer = get_tokenizer()
    long_count = 0

    def encode(inst: Tuple[str, int]) -> OrderedDict:
        text, label = inst
        tokens = tokenizer.tokenize(text)
        max_len = max_seq_length - 2
        if len(tokens) > max_len:
            nonlocal long_count
            long_count = long_count + 1
            if long_count > 10:
                print("long text count", long_count)
        tokens = tokens[:max_len]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        seg_ids = [0] * len(tokens)
        feature: OrderedDict = get_basic_input_feature(tokenizer, max_seq_length, tokens, seg_ids)
        feature['label_ids'] = create_int_feature([label])
        return feature

    return encode


def get_encode_fn_w_data_id(max_seq_length, long_warning=True):
    tokenizer = get_tokenizer()
    long_count = 0

    def encode(inst: TextInstance) -> OrderedDict:
        tokens = tokenizer.tokenize(inst.text)
        max_len = max_seq_length - 2
        if len(tokens) > max_len:
            nonlocal long_count
            long_count = long_count + 1
            if long_count > 10 and long_warning:
                print("long text count", long_count)
        tokens = tokens[:max_len]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        seg_ids = [0] * len(tokens)
        feature: OrderedDict = get_basic_input_feature(tokenizer, max_seq_length, tokens, seg_ids)
        feature['label_ids'] = create_int_feature([inst.label])
        feature['data_id'] = create_int_feature([inst.data_id])
        return feature

    return encode

