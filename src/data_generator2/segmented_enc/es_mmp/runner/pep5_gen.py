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
from data_generator2.segmented_enc.seg_encoder_common import get_random_split_location
from taskman_client.wrapper3 import JobContext


# PWAE
# PairWithAttn -> BothSegPartitionedPair
# PartitionedEncoderIF: BothSegPartitionedPair -> OrderedDict
class PairWithAttnEncoder(PairWithAttnEncoderIF):
    def __init__(self, get_num_delete, tokenizer, partitioned_encoder: PartitionedEncoderIF):
        self.partitioned_encoder: PartitionedEncoderIF = partitioned_encoder
        self.get_num_delete = get_num_delete
        self.tokenizer = tokenizer

    def partition_sel_indices(self, e: Tuple[PairWithAttn, PairWithAttn]) -> \
            Tuple[BothSegPartitionedPair, BothSegPartitionedPair]:
        (pos_pair, pos_attn), (neg_pair, neg_attn) = e

        segment1_s: str = pos_pair.segment1
        assert pos_pair.segment1 == neg_pair.segment1

        segment1_tokens = self.tokenizer.tokenize(segment1_s)
        st, ed = get_random_split_location(segment1_tokens)
        partitioned_segment1: RangePartitionedSegment = RangePartitionedSegment(segment1_tokens, st, ed)

        def partition_pair(pair, attn_score) -> BothSegPartitionedPair:
            segment2_tokens: List[str] = self.tokenizer.tokenize(pair.segment2)
            delete_indices_list = get_delete_indices_for_segment2_inner(attn_score, partitioned_segment1,
                                                                        segment2_tokens, self.get_num_delete)
            partitioned_segment2 = IndicesPartitionedSegment(segment2_tokens, delete_indices_list[0],
                                                             delete_indices_list[1])
            seg_pair = BothSegPartitionedPair(partitioned_segment1, partitioned_segment2, pair)
            return seg_pair

        return partition_pair(pos_pair, pos_attn), partition_pair(neg_pair, neg_attn)

    def encode_fn(self, e: Tuple[PairWithAttn, PairWithAttn]):
        parti_e: Tuple[BothSegPartitionedPair, BothSegPartitionedPair] = self.partition_sel_indices(e)
        a, b = parti_e
        return self.partitioned_encoder.encode_paired(a, b)


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
        generate_train_data(job_no, dataset_name, tfrecord_encoder)


if __name__ == "__main__":
    main()
