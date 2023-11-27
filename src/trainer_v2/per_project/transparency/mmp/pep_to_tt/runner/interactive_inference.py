import numpy as np
import sys
from typing import Tuple, List

import tensorflow as tf

from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.hf_encode_helper import combine_with_sep_cls_and_pad
from tlm.data_gen.base import concat_tuple_windows
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_modeling import PEP_TT_ModelConfig, PEP_TT_Model_Single
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def load_model(model_save_path):
    model_config = PEP_TT_ModelConfig()
    task_model = PEP_TT_Model_Single(model_config)
    task_model.build_model_for_inf(None)

    train_model = task_model.model
    checkpoint = tf.train.Checkpoint(train_model)
    checkpoint.restore(model_save_path).expect_partial()
    return task_model.inf_model


class PEP_TT_Inference:
    def __init__(self, model_config, model_path=None, batch_size=16, model=None):
        self.max_seq_length = model_config.max_seq_length
        c_log.info("Defining network")
        if model is None:
            model = load_model(model_path)

        self.model = model
        self.tokenizer = get_tokenizer()
        self.batch_size = batch_size

    def score_fn(self, qd_list: List[Tuple[str, str]]) -> List[float]:
        def generator():
            for q_term, d_term in qd_list:
                q_term_tokens = self.tokenizer.tokenize(q_term)
                d_term_tokens = self.tokenizer.tokenize(d_term)
                input_ids, segment_ids = combine_with_sep_cls_and_pad(
                    self.tokenizer, q_term_tokens, d_term_tokens, self.max_seq_length)
                yield (input_ids, segment_ids),

        SpecI = tf.TensorSpec([self.max_seq_length], dtype=tf.int32)
        sig = (SpecI, SpecI,),
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=sig)
        dataset = dataset.batch(self.batch_size)
        probs = self.model.predict(dataset)
        return np.reshape(probs, [-1]).tolist()


def main():
    c_log.info(__file__)
    model_path = sys.argv[1]
    strategy = get_strategy()
    model_config = PEP_TT_ModelConfig()
    with strategy.scope():
        inf_helper = PEP_TT_Inference(model_config, model_path)

        while True:
            query = input("Enter query term: ")
            doc = input("Enter document term: ")
            ret = inf_helper.score_fn([(query, doc)])[0]
            print(ret)


if __name__ == "__main__":
    main()
