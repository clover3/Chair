import sys

from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.es_common.partitioned_encoder import PartitionedEncoder, build_get_num_delete_fn
from data_generator2.segmented_enc.es_mmp.pair_w_attn_encoder import PairWithAttnEncoder
from data_generator2.segmented_enc.es_mmp.pep_attn_common import generate_train_data_repeat_pos
from data_generator2.segmented_enc.es_common.pep_attn_common import PairWithAttnEncoderIF
from taskman_client.wrapper3 import JobContext


def main():
    job_no = int(sys.argv[1])
    del_rate = 0.5
    dataset_name = "mmp_pep5"

    with JobContext(f"pep5_gen_{job_no}"):
        get_num_delete = build_get_num_delete_fn(del_rate)

        # Deciding encoder logic
        partition_len = 256
        tokenizer = get_tokenizer()
        partitioned_encoder = PartitionedEncoder(tokenizer, partition_len)
        tfrecord_encoder: PairWithAttnEncoderIF = PairWithAttnEncoder(get_num_delete, tokenizer, partitioned_encoder)
        generate_train_data_repeat_pos(job_no, dataset_name, tfrecord_encoder)


if __name__ == "__main__":
    main()
