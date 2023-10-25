import sys
from typing import List, Tuple

from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.es_common.es_two_seg_common import BothSegPartitionedPair, \
    RangePartitionedSegment, IndicesPartitionedSegment
from data_generator2.segmented_enc.es_common.evidence_selector_by_attn import get_delete_indices_for_segment2_inner
from data_generator2.segmented_enc.es_common.partitioned_encoder import PartitionedEncoder, PartitionedEncoderIF, \
    build_get_num_delete_fn
from data_generator2.segmented_enc.es_mmp.pep_attn_common import generate_train_data
from data_generator2.segmented_enc.es_common.pep_attn_common import PairWithAttn, PairWithAttnEncoderIF
from data_generator2.segmented_enc.es_mmp.runner.pep5_gen import PairWithAttnEncoder
from data_generator2.segmented_enc.seg_encoder_common import get_random_split_location
from taskman_client.wrapper3 import JobContext


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
