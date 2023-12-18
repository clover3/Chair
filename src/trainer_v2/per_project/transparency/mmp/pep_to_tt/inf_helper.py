from typing import List, Tuple

import numpy as np
import tensorflow as tf

from data_generator.tokenizer_wo_tf import get_tokenizer
from data_generator2.segmented_enc.hf_encode_helper import combine_with_sep_cls_and_pad
from trainer_v2.chair_logging import c_log

from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_modeling import PEP_TT_Model_Single, PEP_TT_ModelConfig, \
    PEP_TT_Model, PEP_TT_Model_Single_BERT_Init


def load_model(model_save_path, task_model_cls=PEP_TT_Model_Single):
    model_config = PEP_TT_ModelConfig()
    task_model = task_model_cls(model_config)
    task_model.build_model_for_inf(None)

    train_model = task_model.model
    checkpoint = tf.train.Checkpoint(train_model)
    checkpoint.restore(model_save_path).expect_partial()
    return task_model.inf_model


class PEP_TT_Inference:
    def __init__(self, model_config,
                 model_path=None, batch_size=16, model=None, model_type=None):

        if model_type is None:
            model_type = "PEP_TT_Model_Single"

        model_classes_d = {
            "PEP_TT_Model_Single": PEP_TT_Model_Single,
            "PEP_TT_Model": PEP_TT_Model,
            "PEP_TT_Model_Single_BERT_Init": PEP_TT_Model_Single_BERT_Init
        }
        task_model_cls = model_classes_d[model_type]

        self.max_seq_length = model_config.max_seq_length
        c_log.info("Defining network")
        if model is None:
            model = load_model(model_path, task_model_cls)

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
