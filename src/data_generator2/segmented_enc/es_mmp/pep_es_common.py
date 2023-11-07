import logging
import os
from abc import ABC, abstractmethod
from typing import Iterable, Tuple, List, Callable, OrderedDict

import numpy as np

from cache import load_pickle_from
from cpath import at_output_dir, output_path
from data_generator2.segmented_enc.es_common.es_two_seg_common import Segment1PartitionedPair
from data_generator2.segmented_enc.es_mmp.data_iter_triplets import iter_qd
from misc_lib import exist_or_mkdir, path_join
from tf_util.record_writer_wrap import write_records_w_encode_fn
from trainer_v2.chair_logging import c_log


def iter_es_data(part_no: int) -> Iterable[Tuple[np.array, np.array]]:
    es_save_dir = path_join(output_path, "mmp", "es8")
    batch_no = 0
    while True:
        file_path = path_join(es_save_dir, f"{part_no}_{batch_no}")
        if os.path.exists(file_path):
            c_log.info("Reading %s", file_path)
            obj = load_pickle_from(file_path)
            attn_data_pair: List[Tuple[np.array, np.array]] = obj
            yield from attn_data_pair
        else:
            break
        batch_no += 1


QueryDocES = Tuple[Segment1PartitionedPair, Tuple[np.array, np.array]]


def iter_es_data_pos_neg_pair(part_no: int) -> Iterable[Tuple[QueryDocES, QueryDocES]]:
    es_data_itr = iter_es_data(part_no)
    pairs: Iterable[Segment1PartitionedPair] = iter_qd(part_no)
    paired_iter = iter(zip(pairs, es_data_itr))
    try:
        while True:
            pos_item: QueryDocES = next(paired_iter)
            neg_item: QueryDocES = next(paired_iter)
            yield pos_item, neg_item
    except StopIteration:
        pass


class PairWithESEncoderIF(ABC):
    @abstractmethod
    def encode_fn(self, e: Tuple[QueryDocES, QueryDocES]) -> OrderedDict:
        pass


def generate_train_data(
        job_no: int, dataset_name: str,
        tfrecord_encoder: PairWithESEncoderIF):
    output_dir = at_output_dir("tfrecord", dataset_name)
    exist_or_mkdir(output_dir)
    c_log.setLevel(logging.DEBUG)

    part_no = job_no
    save_path = os.path.join(output_dir, str(part_no))
    c_log.info("Partition %d", part_no)
    data_size = 1000000

    pos_neg_itr: Iterable[Tuple[QueryDocES, QueryDocES]] = iter_es_data_pos_neg_pair(part_no)
    encode_fn: Callable[[Tuple[QueryDocES, QueryDocES]], OrderedDict] = tfrecord_encoder.encode_fn
    write_records_w_encode_fn(save_path, encode_fn, pos_neg_itr, data_size)

