from collections import OrderedDict

from data_generator.create_feature import create_float_feature
from data_generator2.segmented_enc.es_common.es_two_seg_common import BothSegPartitionedPair, RangePartitionedSegment, \
    IndicesPartitionedSegment, PairData
from data_generator2.segmented_enc.es_common.evidence_selector_by_attn import get_delete_indices_for_segment2_inner
from data_generator2.segmented_enc.es_common.partitioned_encoder import PartitionedEncoderIF
from data_generator2.segmented_enc.es_common.pep_attn_common import QDWithAttnEncoderIF
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator

from data_generator2.segmented_enc.seg_encoder_common import get_random_split_location
from trainer_v2.per_project.transparency.mmp.attn_compute.iter_attn import QDWithScoreAttn


class QDWithScoreAttnEncoder(QDWithAttnEncoderIF):
    def __init__(self, get_num_delete, tokenizer, partitioned_encoder: PartitionedEncoderIF):
        self.partitioned_encoder: PartitionedEncoderIF = partitioned_encoder
        self.get_num_delete = get_num_delete
        self.tokenizer = tokenizer

    def encode_fn(self, e: Tuple[QDWithScoreAttn, QDWithScoreAttn]) -> OrderedDict:
        qd_pos, qd_neg = e
        segment1_tokens = self.tokenizer.tokenize(qd_pos.query)
        assert qd_pos.query == qd_neg.query
        st, ed = get_random_split_location(segment1_tokens)
        partitioned_segment1: RangePartitionedSegment = RangePartitionedSegment(segment1_tokens, st, ed)

        def partition_pair(qd: QDWithScoreAttn) -> BothSegPartitionedPair:
            segment2_tokens: List[str] = self.tokenizer.tokenize(qd.doc)
            delete_indices_list = get_delete_indices_for_segment2_inner(qd.attn, partitioned_segment1,
                                                                        segment2_tokens, self.get_num_delete)
            partitioned_segment2 = IndicesPartitionedSegment(segment2_tokens, delete_indices_list[0],
                                                             delete_indices_list[1])
            pair = PairData(qd.query, qd.doc, "1", "none")
            seg_pair = BothSegPartitionedPair(partitioned_segment1, partitioned_segment2, pair)
            return seg_pair

        pos_partitioned: BothSegPartitionedPair = partition_pair(qd_pos)
        neg_partitioned: BothSegPartitionedPair = partition_pair(qd_neg)
        features: OrderedDict = self.partitioned_encoder.encode_paired(pos_partitioned, neg_partitioned)
        features["score1"] = create_float_feature([qd_pos.score])
        features["score2"] = create_float_feature([qd_neg.score])
        return features
