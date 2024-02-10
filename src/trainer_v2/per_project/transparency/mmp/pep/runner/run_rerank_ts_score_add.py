

import sys
from typing import List, Iterable, Callable, Tuple

import tensorflow as tf
from omegaconf import OmegaConf
from omegaconf.errors import ConfigAttributeError

from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.es_common.es_two_seg_common import BothSegPartitionedPair
from data_generator2.segmented_enc.es_common.partitioned_encoder import PartitionedEncoder, PartitionedEncoderIF, \
    PartitionedEncoderCompressMask
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.definitions import ModelConfig512_1
from trainer_v2.custom_loop.neural_network_def.two_seg_alt import CombineByScoreAdd
from trainer_v2.custom_loop.neural_network_def.two_seg_two_model import TwoSegConcatLogitCombineTwoModel
from trainer_v2.per_project.transparency.mmp.eval_helper.rerank_w_conf import run_rerank_with_conf_common
from trainer_v2.per_project.transparency.mmp.pep.pep_rerank import partition_query_new, QDPartitioning
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def get_encode_fn(conf, segment_len, tokenizer) -> Callable[[BothSegPartitionedPair], Tuple] :
    try:
        if conf.partitioner == "compress":
            encoder: PartitionedEncoderIF = PartitionedEncoderCompressMask(tokenizer, segment_len)
            c_log.info("Use PartitionedEncoderCompressMask")
        else:
            encoder: PartitionedEncoderIF = PartitionedEncoder(tokenizer, segment_len)
    except ConfigAttributeError:
        encoder: PartitionedEncoderIF = PartitionedEncoder(tokenizer, segment_len)
    encode_fn: Callable[[BothSegPartitionedPair], Tuple] = encoder.encode_to_ids
    return encode_fn


def load_two_seg_concat_model(conf, model_config):
    task_model = TwoSegConcatLogitCombineTwoModel(model_config, CombineByScoreAdd)
    task_model.build_model(None)
    c_log.info("Loading model from %s", conf.model_path)
    task_model.load_checkpoint(conf.model_path)
    return task_model.point_model


def get_pep_scorer_from_two_model(
        conf,
        batch_size=16,
) -> Callable[[List[Tuple[str, str]]], Iterable[float]]:
    model_config = ModelConfig512_1()
    max_seq_length = model_config.max_seq_length
    segment_len = int(max_seq_length / 2)
    c_log.info("Defining network")
    inference_model = load_two_seg_concat_model(conf, model_config)
    inference_model.summary()

    tokenizer = get_tokenizer()
    partitioner = QDPartitioning(tokenizer)

    encode_fn = get_encode_fn(conf, segment_len, tokenizer)
    SpecI = tf.TensorSpec([max_seq_length], dtype=tf.int32)
    sig = (SpecI, SpecI,),
    is_first_inf = True

    def score_fn(qd_list: List[Tuple[str, str]]):
        def generator():
            for qd in qd_list:
                e: BothSegPartitionedPair = partitioner.partition_query_new(qd)
                ret = encode_fn(e)
                input_ids, segment_ids = ret
                yield (input_ids, segment_ids),

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=sig)
        dataset = dataset.batch(batch_size)
        output = inference_model.predict(dataset)
        nonlocal is_first_inf
        if is_first_inf:
            c_log.info("First batch predict done. %d items", len(output))
            is_first_inf = False
        return output[:, 0]

    return score_fn


def main():
    c_log.info(__file__)
    get_scorer_fn = get_pep_scorer_from_two_model
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    strategy = get_strategy()
    with strategy.scope():
        run_rerank_with_conf_common(conf, get_scorer_fn)


if __name__ == "__main__":
    main()
