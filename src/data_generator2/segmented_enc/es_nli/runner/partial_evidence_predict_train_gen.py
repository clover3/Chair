import os
import pickle
from typing import List

from cpath import at_output_dir
from data_generator2.segmented_enc.es_nli.path_helper import get_evidence_selected0_path
from misc_lib import exist_or_mkdir
from tf_util.record_writer_wrap import write_records_w_encode_fn
from data_generator2.segmented_enc.es_nli.common import PHSegmentedPair, get_ph_segment_pair_encode_fn


def gen_mnli(split):
    output_dir = at_output_dir("tfrecord", "nli_pep1")
    data_size = 400 * 1000 if split == "train" else 10000
    source_path = get_evidence_selected0_path(split)
    exist_or_mkdir(output_dir)
    output_path = os.path.join(output_dir, split)
    payload: List[PHSegmentedPair] = pickle.load(open(source_path, "rb"))

    segment_len = 300
    encode_fn = get_ph_segment_pair_encode_fn(segment_len)
    write_records_w_encode_fn(output_path, encode_fn, payload, data_size)


def main():
    gen_mnli("dev")
    gen_mnli("train")


if __name__ == "__main__":
    main()