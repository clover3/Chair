from collections import OrderedDict
from typing import Tuple, Dict

from transformers import AutoTokenizer

from data_generator.create_feature import create_int_feature



def pad_truncate(seq, max_seq_length):
    seq = seq[:max_seq_length]
    pad_len = max_seq_length - len(seq)
    seq = seq + [0] * pad_len
    return seq



def get_multi_text_encode_fn(max_text_seq_length, n_text):
    def encode_fn(n_items: Tuple[Dict]) -> OrderedDict:
        assert len(n_items) == n_text
        features = OrderedDict()
        for idx, item in enumerate(n_items):
            assert len(item['input_ids']) == len(item['attention_mask'])
            input_ids = pad_truncate(item['input_ids'], max_text_seq_length)
            attention_mask = pad_truncate(item['attention_mask'], max_text_seq_length)
            features[f"input_ids_{idx}"] = create_int_feature(input_ids)
            features[f"attention_mask_{idx}"] = create_int_feature(attention_mask)
        return features

    return encode_fn


def get_transformer_pair_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_pair(pair):
        t1, t2 = pair
        return tokenizer(t1), tokenizer(t2)

    return tokenize_pair