import logging
import os
import sys
from typing import Iterable, Callable, Tuple
from typing import OrderedDict

from cpath import at_output_dir
from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.es_common.partitioned_encoder import PartitionedEncoder, build_get_num_delete_fn
from data_generator2.segmented_enc.es_common.pep_attn_common import PairWithAttn, PairWithAttnEncoderIF
from data_generator2.segmented_enc.es_mmp.pep_attn_common import iter_attention_mmp_pos_neg_paried
from data_generator2.segmented_enc.es_mmp.pair_w_attn_encoder import PairWithAttnEncoder
from misc_lib import exist_or_mkdir
from taskman_client.wrapper3 import JobContext
from tf_util.record_writer_wrap import write_records_w_encode_fn
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.split_iter import get_valid_mmp_partition


def generate_train_data(job_no: int, dataset_name: str, tfrecord_encoder: PairWithAttnEncoderIF):
    output_dir = at_output_dir("tfrecord", dataset_name)
    exist_or_mkdir(output_dir)
    split = "train"
    c_log.setLevel(logging.DEBUG)

    partition_todo = get_valid_mmp_partition(split)
    st = job_no
    ed = st + 10
    for partition_no in range(st, ed):
        if partition_no not in partition_todo:
            continue
        save_path = os.path.join(output_dir, str(partition_no))

        c_log.info("Partition %d", partition_no)
        data_size = 3000
        pos_neg_pair_itr: Iterable[Tuple[PairWithAttn, PairWithAttn]] = iter_attention_mmp_pos_neg_paried(partition_no)
        encode_fn: Callable[[Tuple[PairWithAttn, PairWithAttn]], OrderedDict] = tfrecord_encoder.encode_fn
        write_records_w_encode_fn(save_path, encode_fn, pos_neg_pair_itr, data_size)


def main():
    job_no = int(sys.argv[1])
    del_rate = 0.5
    dataset_name = "mmp_pep6"

    with JobContext(f"pep6_gen_{job_no}"):
        get_num_delete = build_get_num_delete_fn(del_rate)

        # Deciding encoder logic
        partition_len = 256
        tokenizer = get_tokenizer()
        partitioned_encoder = PartitionedEncoder(tokenizer, partition_len)
        tfrecord_encoder: PairWithAttnEncoderIF = PairWithAttnEncoder(get_num_delete, tokenizer, partitioned_encoder)
        generate_train_data(job_no, dataset_name, tfrecord_encoder)


if __name__ == "__main__":
    main()
