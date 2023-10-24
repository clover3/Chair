from typing import Tuple, List, Iterable, Callable

import numpy as np
import tensorflow as tf

from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.es_common.es_two_seg_common import BothSegPartitionedPair, Segment1PartitionedPair, \
    PairData, RangePartitionedSegment
from data_generator2.segmented_enc.es_common.partitioned_encoder import PartitionedEncoder
from data_generator2.segmented_enc.seg_encoder_common import get_random_split_location
from misc_lib import Averager
from trainer_v2.chair_logging import c_log
from trainer_v2.evidence_selector.environment_qd import ConcatMaskStrategyQD
from trainer_v2.evidence_selector.runner_mmp.dataset_fn import SplitStack
from trainer_v2.per_project.transparency.mmp.pep.pep_rerank import partition_query_new


def get_hide_input_ids(mask_strategy, delete_portion, mask_id):
    def hide_input_ids(
            input_ids: np.array,
            segment_ids: np.array,
            scores: np.array):
        is_evidence_mask: np.array = mask_strategy.get_deletable_evidence_mask(input_ids, segment_ids)
        n_evidence = np.count_nonzero(is_evidence_mask)
        n_delete = int(delete_portion * n_evidence)
        neg_large = -99999. + np.min(scores)

        is_not_evidence = np.logical_not(is_evidence_mask)
        scores_masked = scores + neg_large * is_not_evidence.astype(np.float32)
        scores_rank_descending = np.argsort(scores_masked)[::-1]

        delete_mask = np.zeros_like(input_ids, np.int32)
        for i in scores_rank_descending[:n_delete]:
            delete_mask[i] = 1

        select_mask_np = np.logical_not(delete_mask)
        new_input_ids = input_ids * select_mask_np + (1 - select_mask_np) * mask_id

        return new_input_ids
    return hide_input_ids



class HideStrategyMinClip:
    def __init__(self, mask_strategy, init_delete_portion, mask_id=0):
        self.mask_id = mask_id
        self.init_delete_portion = init_delete_portion
        self.mask_strategy = mask_strategy
        self.delete_rate = Averager()
    #
    def hide_input_ids(
            self,
            input_ids: np.array,
            segment_ids: np.array,
            scores: np.array):
        is_evidence_mask: np.array = self.mask_strategy.get_deletable_evidence_mask(input_ids, segment_ids)
        n_delete = self.get_n_delete(input_ids, segment_ids, is_evidence_mask)

        neg_large = -99999. + np.min(scores)

        is_not_evidence = np.logical_not(is_evidence_mask)
        scores_masked = scores + neg_large * is_not_evidence.astype(np.float32)
        scores_rank_descending = np.argsort(scores_masked)[::-1]

        delete_mask = np.zeros_like(input_ids, np.int32)
        for i in scores_rank_descending[:n_delete]:
            delete_mask[i] = 1

        select_mask_np = np.logical_not(delete_mask)
        new_input_ids = input_ids * select_mask_np + (1 - select_mask_np) * self.mask_id

        return new_input_ids

    def get_n_delete(self, input_ids, segment_ids, is_evidence_mask):
        n_evidence = np.count_nonzero(is_evidence_mask)
        query_like_segment_mask = self.mask_strategy.get_query_like_segment_mask(input_ids, segment_ids)
        n_delete = int(self.init_delete_portion * n_evidence)
        n_query_tokens = sum(query_like_segment_mask)
        n_keep = n_query_tokens
        max_delete = n_evidence - n_keep
        n_delete = min(max_delete, n_delete)
        delete_rate = n_delete / n_evidence
        self.delete_rate.append(delete_rate)
        return n_delete


InputIdsSegmentIds = Tuple[List, List]


def build_dataset_from_ids_pair(ids_pair_list, max_seq_length):
    SpecI = tf.TensorSpec([max_seq_length], dtype=tf.int32)
    sig = (SpecI, SpecI)
    def generator() -> Iterable[Tuple[InputIdsSegmentIds]]:
        for ids_pair in ids_pair_list:
            yield ids_pair

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=sig)
    return dataset


def build_evidence_selection_dataset(ids_pair_list, max_seq_length, batch_size):
    dataset = build_dataset_from_ids_pair(ids_pair_list, max_seq_length)
    dataset = dataset.batch(batch_size)
    split_stack_module = SplitStack(max_seq_length)

    def split_stack_apply(*x):
        input_ids, segment_ids = x  # Two segment concatenated
        new_input_ids, new_segment_ids = split_stack_module.apply(input_ids, segment_ids)
        return (new_input_ids, new_segment_ids),

    split_flat_dataset = dataset.map(
        split_stack_apply,
        num_parallel_calls=tf.data.AUTOTUNE)

    return split_flat_dataset


def build_dataset_w_es_score(
        hide_input_id: Callable,
        ids_pair_list: List,
        es_pred: np.array,
        partition_len: int,
        batch_size: int):
    max_seq_length = partition_len * 2
    n_qd = len(ids_pair_list)
    assert n_qd == len(es_pred)

    # Build new dataset
    masked_payload = []
    for i in range(n_qd):
        scores: np.array = es_pred[i]
        input_ids, segment_ids = ids_pair_list[i]
        new_input_ids1, = hide_input_id(
            input_ids[:partition_len], segment_ids[:partition_len], scores[0])
        new_input_ids2, = hide_input_id(
            input_ids[partition_len:], segment_ids[partition_len:], scores[1])
        new_input_ids = np.concat([new_input_ids1, new_input_ids2], axis=0)
        masked_payload.append((new_input_ids, segment_ids))

    masked_dataset = build_dataset_from_ids_pair(masked_payload, max_seq_length)
    masked_dataset = masked_dataset.map(lambda x: (x,), masked_dataset)
    return masked_dataset.batch(batch_size)


def get_pep_scorer_es(
        conf,
        hide_input_id: Callable,
        batch_size=16,
) -> Callable[[List[Tuple[str, str]]], Iterable[float]]:
    model_path = conf.pep_model_path
    partition_len = 256
    max_seq_length = partition_len * 2
    c_log.info("Loading model from %s", model_path)
    inference_model = tf.keras.models.load_model(model_path, compile=False)
    c_log.info("Loading es model from %s", conf.es_model_path)
    es_model = tf.keras.models.load_model(conf.es_model_path, compile=False)

    tokenizer = get_tokenizer()
    encoder = PartitionedEncoder(tokenizer, partition_len)
    encode_fn: Callable[[BothSegPartitionedPair], Tuple] = encoder.encode_to_ids

    def score_fn(qd_list: List[Tuple[str, str]]):
        n_qd = len(qd_list)

        pair_list: List[BothSegPartitionedPair] = [partition_query_new(tokenizer, qd) for qd in qd_list]
        ids_pair_list: List[InputIdsSegmentIds] = [encode_fn(e) for e in pair_list]

        # Build dataset for evidence selection purpose

        es_dataset = build_evidence_selection_dataset(ids_pair_list, max_seq_length, batch_size)
        ret = es_model.predict(es_dataset)
        es_pred = stack_output(ret[:, :, 1])  # [N, 2, segment_len]

        assert n_qd == len(es_pred)

        # Build new dataset
        masked_payload = []
        for i in range(n_qd):
            scores: np.array = es_pred[i]
            input_ids, segment_ids = ids_pair_list[i]
            new_input_ids1 = hide_input_id(
                input_ids[:partition_len], segment_ids[:partition_len], scores[0])
            new_input_ids2 = hide_input_id(
                input_ids[partition_len:], segment_ids[partition_len:], scores[1])
            new_input_ids = np.concatenate([new_input_ids1, new_input_ids2], axis=0)
            masked_payload.append((new_input_ids, segment_ids))

        masked_dataset = build_dataset_from_ids_pair(masked_payload, max_seq_length)

        def pair_map(input_ids, segment_ids):
            return (input_ids, segment_ids),

        masked_dataset = masked_dataset.batch(batch_size)
        masked_dataset = masked_dataset.map(pair_map)
        output = inference_model.predict(masked_dataset)
        return output[:, 1]

    c_log.info("Defining network")
    return score_fn


class PEP_ES_Scorer:
    def __init__(self,
        conf,
        hide_input_id: Callable,
        batch_size=16):
        model_path = conf.pep_model_path
        partition_len = 256
        self.max_seq_length = partition_len * 2
        c_log.info("Loading model from %s", model_path)
        self.inference_model = tf.keras.models.load_model(model_path, compile=False)
        c_log.info("Loading es model from %s", conf.es_model_path)
        self.es_model = tf.keras.models.load_model(conf.es_model_path, compile=False)

        self.tokenizer = get_tokenizer()
        encoder = PartitionedEncoder(self.tokenizer, partition_len)
        self.encode_fn: Callable[[BothSegPartitionedPair], Tuple] = encoder.encode_to_ids

        self.hide_input_id = hide_input_id
        self.batch_size = batch_size
        self.partition_len = partition_len
        self.split_dict = {}

    def apply_segmentation_to_seg1(self, tokenizer, item: PairData) -> Segment1PartitionedPair:
        segment1_tokens = tokenizer.tokenize(item.segment1)
        segment2_tokens: List[str] = tokenizer.tokenize(item.segment2)
        st, ed = get_random_split_location(segment1_tokens)
        segment1 = RangePartitionedSegment(segment1_tokens, st, ed)
        return Segment1PartitionedPair(segment1, segment2_tokens, item)

    def partition_query(self, qd_pair: Tuple[str, str]) -> BothSegPartitionedPair:
        query, document = qd_pair
        pair_data = PairData(query, document, "0", "0")
        segment1_tokens = self.tokenizer.tokenize(pair_data.segment1)

        if query in self.split_dict:
            st, ed = self.split_dict[query]
        else:
            st, ed = get_random_split_location(segment1_tokens)
            self.split_dict[query] = st, ed

        segment2_tokens: List[str] = self.tokenizer.tokenize(pair_data.segment2)
        segment1 = RangePartitionedSegment(segment1_tokens, st, ed)
        pair = Segment1PartitionedPair(segment1, segment2_tokens, pair_data)
        return BothSegPartitionedPair.from_seg1_partitioned_pair(pair)

    def score_fn(self, qd_list: List[Tuple[str, str]]):
        n_qd = len(qd_list)

        pair_list: List[BothSegPartitionedPair] = [self.partition_query(qd) for qd in qd_list]
        ids_pair_list: List[InputIdsSegmentIds] = [self.encode_fn(e) for e in pair_list]

        # Build dataset for evidence selection purpose
        es_dataset = build_evidence_selection_dataset(ids_pair_list, self.max_seq_length, self.batch_size)
        ret = self.es_model.predict(es_dataset)
        es_pred = stack_output(ret[:, :, 1])  # [N, 2, segment_len]

        assert n_qd == len(es_pred)

        # Build new dataset
        masked_payload = []
        for i in range(n_qd):
            scores: np.array = es_pred[i]
            input_ids, segment_ids = ids_pair_list[i]
            partition_len = self.partition_len
            new_input_ids1 = self.hide_input_id(
                input_ids[:partition_len], segment_ids[:partition_len], scores[0])
            new_input_ids2 = self.hide_input_id(
                input_ids[partition_len:], segment_ids[partition_len:], scores[1])
            new_input_ids = np.concatenate([new_input_ids1, new_input_ids2], axis=0)
            masked_payload.append((new_input_ids, segment_ids))

        masked_dataset = build_dataset_from_ids_pair(masked_payload, self.max_seq_length)

        def pair_map(input_ids, segment_ids):
            return (input_ids, segment_ids),

        masked_dataset = masked_dataset.batch(self.batch_size)
        masked_dataset = masked_dataset.map(pair_map)
        output = self.inference_model.predict(masked_dataset)
        return output[:, 1]



def stack_output(output):
    assert len(output) % 2 == 0
    n_inst = int(len(output) / 2)
    return np.reshape(output, [n_inst, 2, -1])


def test_hide_input_ids():
    maks_strategy = ConcatMaskStrategyQD()
    mask_id = 0
    delete_portion = 0.5
    hide_input_ids = get_hide_input_ids(maks_strategy, delete_portion, mask_id)
    input_ids = [101, 7592, 1010, 2088, 102, 1437, 2003, 2307, 1005, 9999, 8888, 2003, 7777, 102]
    segment_ids = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    print(input_ids)
    print(segment_ids)
    scores = np.random.rand(len(input_ids))
    print(scores)
    new_input_ids = hide_input_ids(input_ids, segment_ids, scores)
    print("input_ids", input_ids)
    print("segment_ids", segment_ids)
    print("new_input_ids", new_input_ids.tolist())
    print("scores", scores.tolist())


    if __name__ == "__main__":
        test_hide_input_ids()