from typing import List, Iterable, Callable, Tuple, Dict

from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.es_common.es_two_seg_common import BothSegPartitionedPair, Segment1PartitionedPair, \
    PairData, RangePartitionedSegment
from data_generator2.segmented_enc.es_common.partitioned_encoder import apply_segmentation_to_seg1, \
    get_both_seg_partitioned_to_input_ids, PartitionedEncoder
from data_generator2.segmented_enc.seg_encoder_common import get_random_split_location
from trainer_v2.chair_logging import c_log
import tensorflow as tf


def partition_query_new(
        tokenizer, qd_pair: Tuple[str, str]) -> BothSegPartitionedPair:
    query, document = qd_pair
    pair_data = PairData(query, document, "0", "0")
    pair: Segment1PartitionedPair = apply_segmentation_to_seg1(tokenizer, pair_data)
    ph_seg_pair = BothSegPartitionedPair.from_seg1_partitioned_pair(pair)
    return ph_seg_pair


class QDPartitioning:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.q_cache: Dict[str, RangePartitionedSegment] = {}

    def partition_query_new(self, qd_pair: Tuple[str, str]) -> BothSegPartitionedPair:
        query, document = qd_pair
        if query in self.q_cache:
            segment1 = self.q_cache[query]
        else:
            segment1_tokens = self.tokenizer.tokenize(query)
            st, ed = get_random_split_location(segment1_tokens)
            segment1 = RangePartitionedSegment(segment1_tokens, st, ed)
            self.q_cache[query] = segment1

        pair_data = PairData(query, document, "0", "0")
        segment2_tokens: List[str] = self.tokenizer.tokenize(document)
        pair = Segment1PartitionedPair(segment1, segment2_tokens, pair_data)
        ph_seg_pair = BothSegPartitionedPair.from_seg1_partitioned_pair(pair)
        return ph_seg_pair


def get_pep_scorer_from_pointwise(
        conf,
        batch_size=16,
) -> Callable[[List[Tuple[str, str]]], Iterable[float]]:
    model_path = conf.model_path
    segment_len = 256
    max_seq_length = segment_len * 2
    c_log.info("Loading model from %s", model_path)
    pointwise_model = tf.keras.models.load_model(model_path, compile=False)
    inference_model = pointwise_model
    tokenizer = get_tokenizer()
    encoder = PartitionedEncoder(tokenizer, segment_len)
    encode_fn: Callable[[BothSegPartitionedPair], Tuple] = encoder.encode_to_ids
    SpecI = tf.TensorSpec([max_seq_length], dtype=tf.int32)
    sig = (SpecI, SpecI,),

    def score_fn(qd_list: List[Tuple[str, str]]):
        def generator():
            for qd in qd_list:
                e: BothSegPartitionedPair = partition_query_new(tokenizer, qd)
                ret = encode_fn(e)
                input_ids, segment_ids = ret
                yield (input_ids, segment_ids),

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=sig)
        dataset = dataset.batch(batch_size)
        output = inference_model.predict(dataset)
        return output[:, 1]

    c_log.info("Defining network")
    return score_fn

