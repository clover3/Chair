import os
import pickle
from collections import OrderedDict
from typing import List, Tuple

from cpath import at_output_dir
from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.es.path_helper import get_evidence_selected0_path
from data_generator2.segmented_enc.segmented_tfrecord_gen import encode_triplet
from misc_lib import exist_or_mkdir
from tf_util.record_writer_wrap import write_records_w_encode_fn
from tlm.data_gen.base import combine_with_sep_cls, get_basic_input_feature_as_list, concat_triplet_windows
from trainer_v2.custom_loop.attention_helper.evidence_selector_0 import SegmentedPair2


def gen_mnli(split):
    output_dir = at_output_dir("tfrecord", "nli_pep1")
    data_size = 400 * 1000 if split == "train" else 10000
    source_path = get_evidence_selected0_path(split)
    exist_or_mkdir(output_dir)
    output_path = os.path.join(output_dir, split)
    payload: List[SegmentedPair2] = pickle.load(open(source_path, "rb"))

    segment_len = 300
    tokenizer = get_tokenizer()

    def encode_fn(e: SegmentedPair2) -> OrderedDict:
        triplet_list = []
        for i in [0, 1]:
            tokens, segment_ids = combine_with_sep_cls(segment_len, e.get_partial_prem(i), e.get_partial_hypo(i))
            triplet = get_basic_input_feature_as_list(tokenizer, segment_len,
                                                      tokens, segment_ids)
            triplet_list.append(triplet)
        triplet = concat_triplet_windows(triplet_list, segment_len)
        return encode_triplet(triplet, e.nli_pair.get_label_as_int())

    write_records_w_encode_fn(output_path, encode_fn, payload, data_size)


def main():
    gen_mnli("dev")
    gen_mnli("train")


if __name__ == "__main__":
    main()