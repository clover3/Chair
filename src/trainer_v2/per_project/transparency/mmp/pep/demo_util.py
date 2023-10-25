from typing import Callable, Tuple, List, Iterable

import tensorflow as tf

from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.hf_encode_helper import combine_with_sep_cls_and_pad
from tlm.data_gen.base import concat_tuple_windows
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.definitions import ModelConfig512_1
from trainer_v2.custom_loop.neural_network_def.two_seg_alt import CombineByScoreAdd
from trainer_v2.custom_loop.neural_network_def.two_seg_two_model import TwoSegConcatLogitCombineTwoModel
from trainer_v2.custom_loop.per_task.ts_util import load_local_decision_model


def get_pep_local_decision(
        model_path,
        batch_size=16,
) -> Callable[[Tuple[List[str], List[str]]], Iterable[float]]:
    model_config = ModelConfig512_1()
    max_seq_length = model_config.max_seq_length
    segment_len = int(max_seq_length / 2)
    partition_len = segment_len
    # inference_model = load_two_seg_concat_model(conf, model_config)
    model: tf.keras.models.Model = load_local_decision_model(
        model_config, model_path)

    tokenizer = get_tokenizer()
    SpecI = tf.TensorSpec([max_seq_length], dtype=tf.int32)
    sig = (SpecI, SpecI,),

    def encode_to_ids(seg_pair_pair):
        tuple_list = []
        for partial_seg1, partial_seg2 in seg_pair_pair:
            input_ids, segment_ids = combine_with_sep_cls_and_pad(
                tokenizer, partial_seg1, partial_seg2, partition_len)
            tuple_list.append((input_ids, segment_ids))
        return concat_tuple_windows(tuple_list, partition_len)

    def score_fn(qd: Tuple[List[str], List[str]]):
        def generator():
            dummy_pair = ([], [])
            seg_pair_pair = (qd, dummy_pair)
            input_ids, segment_ids = encode_to_ids(seg_pair_pair)
            yield (input_ids, segment_ids),

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=sig)
        dataset = dataset.batch(batch_size)
        output = model.predict(dataset)
        return output

    c_log.info("Defining network")
    return score_fn


def load_two_seg_concat_model(conf, model_config):
    task_model = TwoSegConcatLogitCombineTwoModel(model_config, CombineByScoreAdd)
    task_model.build_model(None)
    c_log.info("Loading model from %s", conf.model_path)
    task_model.load_checkpoint(conf.model_path)
    return task_model.point_model