from typing import List, Tuple

import tensorflow as tf
from transformers import AutoTokenizer

from data_generator.common import get_tokenizer
from data_generator2.segmented_enc.hf_encode_helper import combine_with_sep_cls_and_pad
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.definitions import ModelConfigType
from trainer_v2.per_project.transparency.mmp.pep_short.pep_short_modeling import PEPShortModelConfig, PEP_TTShort
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def load_ts_concat_local_decision_model(
        new_model_config: ModelConfigType, model_save_path):
    task_model = PEP_TTShort(
        new_model_config)
    task_model.build_model_for_inf(None)
    checkpoint = tf.train.Checkpoint(task_model.model)
    c_log.info("Loading model from %s", model_save_path)
    checkpoint.restore(model_save_path).expect_partial()
    task_model.inf_model.summary()
    return task_model.inf_model


class InfHelper:
    def __init__(self, model_config, batch_size=16, model=None):
        self.max_seq_length = model_config.max_seq_length
        self.tokenizer = get_tokenizer()
        self.model = model
        self.batch_size = batch_size

    def score_fn(self, qd_list: List[Tuple[List[str], List[str]]]) -> List[float]:
        def generator():
            for qd in qd_list:
                q, d = qd
                input_ids, segment_ids = combine_with_sep_cls_and_pad(
                    self.tokenizer, q, d, self.max_seq_length)

                yield (input_ids, segment_ids),

        SpecI = tf.TensorSpec([self.max_seq_length], dtype=tf.int32)
        sig = (SpecI, SpecI,),
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=sig)
        dataset = dataset.batch(self.batch_size)
        output = self.model.predict(dataset)
        return output[:, 0]


def get_term_pair_predictor16(
        model_path,
):
    strategy = get_strategy()
    with strategy.scope():
        model_config = PEPShortModelConfig()
        model = load_ts_concat_local_decision_model(model_config, model_path)
        pep = InfHelper(model_config, model=model)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    def score_term_pairs(term_pairs: List[Tuple[str, str]]):
        payload = []
        info = []
        for q_term, d_term in term_pairs:
            q_tokens = tokenizer.tokenize(q_term)
            d_tokens = tokenizer.tokenize(d_term)
            info.append((q_term, d_term))
            payload.append((q_tokens, d_tokens))

        scores: List[float] = pep.score_fn(payload)
        return scores

    return score_term_pairs
