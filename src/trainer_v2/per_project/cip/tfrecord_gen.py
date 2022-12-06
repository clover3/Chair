import itertools
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Iterator, NamedTuple
from typing import List, Callable, Tuple

import numpy as np

from data_generator.special_tokens import CLS_ID, SEP_ID
from misc_lib import TELI
from tf_util.record_writer_wrap import write_records_w_encode_fn
from tlm.data_gen.bert_data_gen import create_int_feature
from tlm.data_gen.rank_common import join_two_input_ids
from trainer_v2.per_project.cip.cip_common import Comparison, split_into_two
from trainer_v2.per_project.cip.path_helper import get_cip_dataset_path
from trainer_v2.per_project.cip.precomputed_cip import iter_cip_preds


def parse_fail(comparison: Comparison):
    full_pred = np.argmax(comparison.full_pred_probs)
    fail_items: List[Tuple[int, int]] = []
    suc_items: List[Tuple[int, int]] = []

    for ts_probs, info in zip(comparison.ts_pred_probs, comparison.ts_input_info_list):
        ts_probs: Tuple[List[List[float]], List[float]] = ts_probs
        local_d, global_d = ts_probs
        ts_global_pred = np.argmax(global_d)
        if full_pred != ts_global_pred and full_pred == comparison.label:
            fail_items.append(info)
        else:
            suc_items.append(info)

    return fail_items, suc_items


class LabeledInstance(NamedTuple):
    comparison: Comparison
    st: int
    ed: int
    label: int


class ItemSelector(ABC):
    @abstractmethod
    def select(self, itr: Iterator[Comparison]) -> Iterator[LabeledInstance]:
        pass

    def size_guess(self, n_src) -> int:
        return 0


class SelectOneToOne(ItemSelector):
    # Select only for instance with failure
    # Select n positive and n negative
    def select(self, itr: Iterator[Comparison]) -> Iterator[LabeledInstance]:
        for e in itr:
            fail_item, suc_item = parse_fail(e)

            if not fail_item:
                continue

            n_inst = min(len(fail_item), len(suc_item))
            for i in range(n_inst):
                st, ed = fail_item[i]
                yield LabeledInstance(e, st, ed, 1)
                st, ed = suc_item[i]
                yield LabeledInstance(e, st, ed, 0)


class SelectUpToK(ItemSelector):
    def __init__(self, k):
        self.k = k
    # Select only for instance with failure
    # Select n positive and n negative
    def select(self, itr: Iterator[Comparison]) -> Iterator[LabeledInstance]:
        n_fail_so_far = 0
        n_suc_so_far = 0

        for e in itr:
            fail_item, suc_item = parse_fail(e)

            for st, ed in fail_item:
                yield LabeledInstance(e, st, ed, 1)
                n_fail_so_far += 1

            for st, ed in suc_item:
                if  n_suc_so_far < n_fail_so_far * self.k:
                    yield LabeledInstance(e, st, ed, 0)
                    n_suc_so_far += 1

        print("Selected {} fail item (Pos) {} suc item (Neg)".format(n_fail_so_far, n_suc_so_far))


class SelectAll(ItemSelector):
    # Select only for instance with failure
    # Select n positive and n negative
    def select(self, itr: Iterator[Comparison]) -> Iterator[LabeledInstance]:
        n_fail_so_far = 0
        n_suc_so_far = 0

        for e in itr:
            fail_item, suc_item = parse_fail(e)
            for st, ed in fail_item:
                yield LabeledInstance(e, st, ed, 1)
                n_fail_so_far += 1

            for st, ed in suc_item:
                yield LabeledInstance(e, st, ed, 0)
                n_suc_so_far += 1

        print("Selected {} fail item (Pos) {} suc item (Neg)".format(n_fail_so_far, n_suc_so_far))


def count_iter(itr: Iterator, name) -> Iterator:
    cnt = 0
    for e in itr:
        cnt += 1
        yield e
    print("{} has {} itmes".format(name, cnt))


def build_encoded(dataset_name,
                  item_selector: ItemSelector,
                  encode_fn: Callable[[LabeledInstance], OrderedDict]):
    k_validation = 4000
    iter: Iterator[Comparison] = iter_cip_preds()
    val_itr = itertools.islice(iter, k_validation)
    train_itr = itertools.islice(iter, k_validation, None)
    n_train_itr_size = 384702

    todo = [
        ("train_val", val_itr, k_validation),
        ("train", train_itr, n_train_itr_size)
    ]
    for split, itr, src_size in todo:
        labeled_instance_itr: Iterator[LabeledInstance] = item_selector.select(itr)
        labeled_instance_itr = count_iter(labeled_instance_itr, split)
        labeled_instance_itr = TELI(labeled_instance_itr, src_size)
        save_path = get_cip_dataset_path(dataset_name, split)
        out_n = item_selector.size_guess(src_size)
        write_records_w_encode_fn(save_path, encode_fn, labeled_instance_itr, out_n)


def pad_to_length(seq, pad_len):
    seq = seq[:pad_len]
    n_pad = pad_len - len(seq)
    return seq + [0] * n_pad


def encode_together(seq_length, e: LabeledInstance) -> OrderedDict:
    hypo: List[int] = e.comparison.hypo
    input_ids = pad_to_length(hypo, seq_length)
    input_mask = pad_to_length([1] * len(hypo), seq_length)
    segment_ids = [0] * e.st + [1] * (e.ed - e.st) + [0] * (len(hypo) - e.ed)
    segment_ids = pad_to_length(segment_ids, seq_length)
    features = OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features['label_ids'] = create_int_feature([e.label])
    return features


def encode_separate(seq_length, e: LabeledInstance) -> OrderedDict:
    hypo: List[int] = e.comparison.hypo
    hypo1, hypo2 = split_into_two(hypo, e.st, e.ed)
    pair = join_two_input_ids(hypo1, hypo2)
    input_ids = pad_to_length(pair.input_ids, seq_length)
    input_mask = pad_to_length([1] * len(pair.input_ids), seq_length)
    segment_ids = pad_to_length(pair.seg_ids, seq_length)
    features = OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features['label_ids'] = create_int_feature([e.label])
    return features



def encode_three(seq_length, e: LabeledInstance) -> OrderedDict:
    hypo: List[int] = e.comparison.hypo
    hypo1, hypo2 = split_into_two(hypo, e.st, e.ed)
    input_ids = [CLS_ID] + hypo + [SEP_ID] + hypo1 + [SEP_ID] + hypo2 + [SEP_ID]
    segment_ids = [0] * (len(hypo) + 2) + [1] * (len(hypo1) + 1) + [2] * (len(hypo2) + 1)
    input_ids = pad_to_length(input_ids, seq_length)
    input_mask = pad_to_length([1] * len(input_ids), seq_length)
    segment_ids = pad_to_length(segment_ids, seq_length)
    features = OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features['label_ids'] = create_int_feature([e.label])
    return features


