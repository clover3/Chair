import logging
import os
import pickle
from typing import Iterable, Tuple, List, Callable, OrderedDict

import numpy as np

from adhoc.misc_helper import group_pos_neg, enumerate_pos_neg_pairs, enumerate_pos_neg_pairs_once
from cpath import at_output_dir, output_path
from data_generator2.segmented_enc.es_common.es_two_seg_common import PairData
from data_generator2.segmented_enc.es_common.pep_attn_common import PairWithAttn, PairWithAttnEncoderIF
from list_lib import flatten
from misc_lib import exist_or_mkdir, path_join, group_iter
from tf_util.record_writer_wrap import write_records_w_encode_fn
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.split_iter import get_valid_mmp_partition


def iter_attention_data_pair(partition_no) -> Iterable[Tuple[PairData, np.array]]:
    attn_save_dir = path_join(output_path, "msmarco", "passage", "mmp1_attn")
    batch_no = 0
    while True:
        file_path = path_join(attn_save_dir, f"{partition_no}_{batch_no}")
        if os.path.exists(file_path):
            c_log.info("Reading %s", file_path)
            f = open(file_path, "rb")
            obj = CustomUnpickler(f).load()
            attn_data_pair: List[Tuple[PairData, np.array]] = obj
            yield from attn_data_pair
        else:
            break
        batch_no += 1


def generate_train_data(job_no: int, dataset_name: str, tfrecord_encoder: PairWithAttnEncoderIF):
    output_dir = at_output_dir("tfrecord", dataset_name)
    exist_or_mkdir(output_dir)
    split = "train"
    c_log.setLevel(logging.DEBUG)

    partition_todo = get_valid_mmp_partition(split)
    st = job_no
    ed = st + 1
    for partition_no in range(st, ed):
        if partition_no not in partition_todo:
            continue
        save_path = os.path.join(output_dir, str(partition_no))

        c_log.info("Partition %d", partition_no)
        data_size = 30000
        attn_data_pair: Iterable[PairWithAttn] = iter_attention_data_pair(partition_no)
        grouped_itr: Iterable[List[PairWithAttn]] = group_iter(attn_data_pair, get_pair_key)
        pos_neg_itr: Iterable[Tuple[List[PairWithAttn], List[PairWithAttn]]] = map(
            lambda e: group_pos_neg(e, is_pos), grouped_itr)
        pos_neg_pair_itr: Iterable[Tuple[PairWithAttn, PairWithAttn]] = flatten(map(
            enumerate_pos_neg_pairs, pos_neg_itr))

        encode_fn: Callable[[Tuple[PairWithAttn, PairWithAttn]], OrderedDict] = tfrecord_encoder.encode_fn
        write_records_w_encode_fn(save_path, encode_fn, pos_neg_pair_itr, data_size)



class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'data_generator2.segmented_enc.es_two_seg_common':
            module = 'data_generator2.segmented_enc.es_common.es_two_seg_common'
        return super().find_class(module, name)


def get_pair_key(pair_with_attn: PairWithAttn) -> str:
    pair, attn = pair_with_attn
    return pair.segment1


def is_pos(e: PairWithAttn):
    pair, attn = e
    return pair.label == "1"


def iter_attention_mmp_pos_neg_paried(partition_no):
    attn_data_pair: Iterable[PairWithAttn] = iter_attention_data_pair(partition_no)
    grouped_itr: Iterable[List[PairWithAttn]] = group_iter(attn_data_pair, get_pair_key)
    pos_neg_itr: Iterable[Tuple[List[PairWithAttn], List[PairWithAttn]]] = map(
        lambda e: group_pos_neg(e, is_pos), grouped_itr)
    pos_neg_pair_itr: Iterable[Tuple[PairWithAttn, PairWithAttn]] = flatten(map(
        enumerate_pos_neg_pairs_once, pos_neg_itr))
    return pos_neg_pair_itr