import random
from collections import OrderedDict
from typing import Tuple, List

from transformers import AutoTokenizer

from cpath import output_path
from data_generator.create_feature import create_int_feature, create_float_feature
from misc_lib import group_by, get_first, path_join
from tf_util.record_writer_wrap import write_records_w_encode_fn
from trainer_v2.per_project.transparency.misc_common import read_term_pair_table


def get_encode_fn():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    seen = set()
    max_length = 1
    def encode_term(term):
        tokens = tokenizer.tokenize(term)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        if len(tokens) > 1:
            if term not in seen:
                print(f"Term {term} has {len(tokens)} tokens")
                seen.add(term)
        input_ids = input_ids[:max_length]
        pad_len = max_length - len(input_ids)
        return input_ids + [0] * pad_len

    def encode_fn(term_gain: Tuple[str, str, float]):
        q_term, d_term, raw_label = term_gain
        feature: OrderedDict = OrderedDict()
        feature[f"q_term"] = create_int_feature(encode_term(q_term))
        feature[f"d_term"] = create_int_feature(encode_term(d_term))
        feature[f"raw_label"] = create_float_feature([raw_label])
        return feature
    return encode_fn


def read_score_generate_term_pair_score_tfrecord(save_name, score_path):
    term_gain = read_term_pair_table(score_path)
    generate_train_data_inner(save_name, term_gain)


def generate_train_data_inner(
        save_name, term_gain: List[Tuple[str, str, float]]):
    grouped = group_by(term_gain, get_first)
    q_term_list = list(grouped.keys())
    random.shuffle(q_term_list)
    train_portion = 0.8
    val_portion = 0.1
    train_size = int(len(q_term_list) * train_portion)
    val_size = int(len(q_term_list) * val_portion)
    todo = [
        ('train', q_term_list[:train_size]),
        ('val', q_term_list[train_size:train_size + val_size]),
        ('test', q_term_list[train_size + val_size:])
    ]
    encode_fn = get_encode_fn()
    for split, q_term_list_split in todo:
        data = []
        for k in q_term_list_split:
            data.extend(grouped[k])
        save_path = path_join(output_path, "msmarco", "passage", "galign_v2", save_name, split)
        print(f"split {split} has {len(data)} items from {len(q_term_list_split)} q_terms")
        write_records_w_encode_fn(save_path, encode_fn, data)