import logging
import os
import sys
from typing import Iterable, Callable, Tuple, OrderedDict

from cpath import at_output_dir
from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.es_common.partitioned_encoder import PartitionedEncoder, build_get_num_delete_fn
from data_generator2.segmented_enc.es_common.pep_attn_common import QDWithAttnEncoderIF
from data_generator2.segmented_enc.es_mmp.qd_pair_w_attn import QDWithScoreAttnEncoder
from dataset_specific.msmarco.passage.path_helper import train_triples_small_partition_iter
from misc_lib import exist_or_mkdir, path_join
from ptorch.cross_encoder.attention_extractor import AttentionExtractor
from taskman_client.wrapper3 import JobContext
from tf_util.record_writer_wrap import write_records_w_encode_fn
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.attn_compute.iter_attn import iter_attention_data_pair_as_pos_neg, \
    QDWithScoreAttn, get_attn2_save_dir, reshape_flat_pos_neg
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.split_iter import get_valid_mmp_partition


def generate_train_data(job_no: int, dataset_name: str, tfrecord_encoder: QDWithAttnEncoderIF):
    output_dir = at_output_dir("tfrecord", dataset_name)
    save_path = path_join(output_dir, str(job_no))
    exist_or_mkdir(output_dir)
    extractor = AttentionExtractor()

    data_size = 100000
    triplet_itr = train_triples_small_partition_iter(job_no)
    c_log.info("job_no %d", job_no)

    def qd_iter():
        for q, dp, dn in triplet_itr:
            yield q, dp
            yield q, dn

    attn_itr: Iterable[Tuple] = extractor.predict_itr(qd_iter())
    qd_with_attn_itr = (QDWithScoreAttn(*row) for row in attn_itr)
    pn_w_attn_itr: Iterable[Tuple[QDWithScoreAttn, QDWithScoreAttn]] = reshape_flat_pos_neg(qd_with_attn_itr)
    encode_fn: Callable[[Tuple[QDWithScoreAttn, QDWithScoreAttn]], OrderedDict] = tfrecord_encoder.encode_fn
    write_records_w_encode_fn(save_path, encode_fn, pn_w_attn_itr, data_size)


def main():
    job_no = int(sys.argv[1])
    del_rate = 0.5
    dataset_name = "mmp_pep8"
    c_log.setLevel(logging.INFO)

    with JobContext(f"pep8_gen_{job_no}"):
        get_num_delete = build_get_num_delete_fn(del_rate)
        # Deciding encoder logic
        partition_len = 256
        tokenizer = get_tokenizer()
        partitioned_encoder = PartitionedEncoder(tokenizer, partition_len)
        tfrecord_encoder: QDWithAttnEncoderIF = QDWithScoreAttnEncoder(
            get_num_delete, tokenizer, partitioned_encoder)
        generate_train_data(job_no, dataset_name, tfrecord_encoder)


if __name__ == "__main__":
    main()
