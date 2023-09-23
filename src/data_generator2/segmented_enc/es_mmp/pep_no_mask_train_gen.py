import logging
import os
import pickle
from collections import OrderedDict
from typing import List, Iterable

from transformers import AutoTokenizer

from cpath import at_output_dir
from data_generator2.segmented_enc.es_mmp.pep_train_gen import get_ph_segment_pair_encode_fn
from data_generator2.segmented_enc.es_nli.path_helper import get_mmp_es0_path
from data_generator2.segmented_enc.hf_encode_helper import encode_pair, \
    combine_with_sep_cls_and_pad
from misc_lib import exist_or_mkdir, TELI
from tf_util.record_writer_wrap import write_records_w_encode_fn
from data_generator2.segmented_enc.es_nli.common import PHSegmentedPair
from tlm.data_gen.base import concat_tuple_windows
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.split_iter import get_valid_mmp_partition


def remove_deletion(e: PHSegmentedPair):
    return PHSegmentedPair(
        e.p_tokens, e.h_tokens, e.h_st, e.h_ed,
        [], [],
        e.nli_pair
    )

def iterate_ph_segmented_pair(partition_no) -> Iterable[PHSegmentedPair]:
    source_path = get_mmp_es0_path(partition_no)
    c_log.info("Loading pickle from %s", source_path)
    payload: List[PHSegmentedPair] = pickle.load(open(source_path, "rb"))
    payload = [remove_deletion(p) for p in payload]
    return payload


def generate_train_data():
    output_dir = at_output_dir("tfrecord", "mmp_pep2")
    split = "train"
    c_log.setLevel(logging.DEBUG)
    exist_or_mkdir(output_dir)
    segment_len = 256
    encode_fn = get_ph_segment_pair_encode_fn(segment_len)

    for partition_no in get_valid_mmp_partition(split):
        c_log.info("Partition %d", partition_no)
        payload = iterate_ph_segmented_pair(partition_no)
        c_log.info("%d items", len(payload))
        itr = TELI(payload, len(payload))
        output_path = os.path.join(output_dir, str(partition_no))
        write_records_w_encode_fn(output_path, encode_fn, itr)


def main():
    generate_train_data()


if __name__ == "__main__":
    main()