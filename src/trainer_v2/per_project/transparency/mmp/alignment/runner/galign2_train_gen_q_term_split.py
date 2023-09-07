import random
from collections import OrderedDict, Counter
from cpath import output_path
from table_lib import tsv_iter
from misc_lib import path_join, get_second, pick1, group_by, get_first

from transformers import AutoTokenizer
from typing import List, Iterable, Callable, Dict, Tuple, Set
from data_generator.create_feature import create_int_feature, create_float_feature
from tf_util.record_writer_wrap import write_records_w_encode_fn


def get_encode_fn():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    max_length = 1
    def encode_term(term):
        tokens = tokenizer.tokenize(term)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
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


def generate_train_data(save_name, score_path):
    itr = tsv_iter(score_path)
    term_gain: List[Tuple[str, str, float]] = []
    for row in itr:
        qt, dt, score = row
        term_gain.append((qt, dt, float(score)))

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


def main2():
    score_path = path_join(
        output_path, "msmarco", "passage", "align_scores", "candidate2.tsv")
    save_name = "cand2_2"

    generate_train_data(save_name, score_path)


if __name__ == "__main__":
    main2()
