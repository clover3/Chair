from data_generator2.segmented_enc.es_mmp.pep_attn_common import iter_attention_data_pair, is_pos, get_pair_key
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.split_iter import get_valid_mmp_partition
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator
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


def generate_train_data_repeat_pos_itr(job_no: int, dataset_name: str, tfrecord_encoder: PairWithAttnEncoderIF):
    split = "train"
    partition_todo = get_valid_mmp_partition(split)
    st = job_no
    ed = st + 1
    for partition_no in range(st, ed):
        if partition_no not in partition_todo:
            continue

        c_log.info("Partition %d", partition_no)
        data_size = 30000
        attn_data_pair: Iterable[PairWithAttn] = iter_attention_data_pair(partition_no)
        grouped_itr: Iterable[List[PairWithAttn]] = group_iter(attn_data_pair, get_pair_key)
        pos_neg_itr: Iterable[Tuple[List[PairWithAttn], List[PairWithAttn]]] = map(
            lambda e: group_pos_neg(e, is_pos), grouped_itr)
        pos_neg_pair_itr: Iterable[Tuple[PairWithAttn, PairWithAttn]] = flatten(map(
            enumerate_pos_neg_pairs, pos_neg_itr))



def main():
    return NotImplemented


if __name__ == "__main__":
    main()