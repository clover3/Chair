import logging
import os

from cpath import at_output_dir
from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.es_mmp.pep_ph_based_common import get_ph_segment_pair_encode_fn, load_ph_segmented_pair
from misc_lib import exist_or_mkdir, TELI
from tf_util.record_writer_wrap import write_records_w_encode_fn
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.split_iter import get_valid_mmp_partition


def generate_train_data():
    output_dir = at_output_dir("tfrecord", "mmp_pep1")
    split = "train"
    c_log.setLevel(logging.DEBUG)
    exist_or_mkdir(output_dir)
    segment_len = 256
    tokenizer = get_tokenizer()
    encode_fn = get_ph_segment_pair_encode_fn(tokenizer, segment_len)

    for partition_no in get_valid_mmp_partition(split):
        c_log.info("Partition %d", partition_no)
        payload = load_ph_segmented_pair(partition_no)
        c_log.info("%d items", len(payload))
        itr = TELI(payload, len(payload))
        output_path = os.path.join(output_dir, str(partition_no))
        write_records_w_encode_fn(output_path, encode_fn, itr)


def main():
    generate_train_data()


if __name__ == "__main__":
    main()