import time
from collections import OrderedDict

from cache import load_pickle_from
from cpath import output_path
from data_generator.create_feature import create_int_feature, create_float_feature
from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.mnli_common import mnli_encode_common
from dataset_specific.mnli.mnli_reader import MNLIReader, NLIPairData
from list_lib import lflatten
from misc_lib import path_join
from typing import List, Iterable, Callable, Dict, Tuple, Set
import numpy as np

from tlm.data_gen.base import combine_with_sep_cls, get_basic_input_feature_as_list


def load_tli_prediction_train() -> Dict[Tuple[str, str], np.array]:
    tli_dict: Dict[Tuple[str, str], np.array] = {}
    missing = []
    for i in range(384):
        try:
            save_path = path_join(output_path, "tli", "nli_train_pred", str(i))
            d = load_pickle_from(save_path)
            tli_dict.update(d)
        except FileNotFoundError as e:
            print(i)
            missing.append(i)
            pass
    print("Missing jobs: {}".format(missing))
    return tli_dict


def load_tli_prediction_dev() -> Dict[Tuple[str, str], np.array]:
    save_path = path_join(output_path, "tli", "nli_dev_pred")
    tli_dict = load_pickle_from(save_path)
    return tli_dict


def load_tli_prediction(split):
    if split == "train":
        return load_tli_prediction_train()
    elif split == "dev":
        return load_tli_prediction_dev()
    else:
        raise KeyError()


def get_encode_fn(max_seq_length, tli_dims):
    tokenizer = get_tokenizer()

    def entry_encode(pair: NLIPairData, tli_scores: np.array) -> Dict:
        features = OrderedDict()
        p_tokens = tokenizer.tokenize(pair.premise)
        h_sp_tokens = pair.hypothesis.split()
        h_sb_tokens_list = list(map(tokenizer.tokenize, h_sp_tokens))
        h_tokens = lflatten(h_sb_tokens_list)
        assert len(tli_scores) == len(h_sp_tokens)
        prefix_len = len(p_tokens) + 2

        tli_label_table = []
        for label_idx in range(tli_dims):
            tli_label_row = [0] * prefix_len
            for j, sb_tokens in enumerate(h_sb_tokens_list):
                label = tli_scores[j, label_idx]
                for _ in sb_tokens:
                    tli_label_row.append(label)

            pad_len = max_seq_length - len(tli_label_row)
            tli_label_row += [0] * pad_len
            tli_label_row = tli_label_row[:max_seq_length]
            tli_label_table.append(tli_label_row)

        tli_label_flat = lflatten(tli_label_table)
        tokens, segment_ids = combine_with_sep_cls(max_seq_length, p_tokens, h_tokens)
        input_ids, input_mask, segment_ids = get_basic_input_feature_as_list(tokenizer, max_seq_length, tokens, segment_ids)
        features["input_ids"] = create_int_feature(input_ids)
        features["input_mask"] = create_int_feature(input_mask)
        features["segment_ids"] = create_int_feature(segment_ids)
        features['label_ids'] = create_int_feature([pair.get_label_as_int()])
        features['tli_label'] = create_float_feature(tli_label_flat)
        return features

    return entry_encode


def do_for_split(split):
    tli_dict: Dict[Tuple[str, str], np.array] = load_tli_prediction(split)
    tli_dims = 3
    max_seq_length = 300
    encode_fn = get_encode_fn(max_seq_length, tli_dims)

    def encode_fn_wrap(pair: NLIPairData):
        key = pair.premise, pair.hypothesis
        tli_scores = tli_dict[key]
        return encode_fn(pair, tli_scores)

    save_path = path_join(output_path, "tfrecord", "mnli_tli_train", split)
    mnli_encode_common(encode_fn_wrap, split, save_path)


def main():
    do_for_split("train")
    do_for_split("dev")


if __name__ == "__main__":
    main()