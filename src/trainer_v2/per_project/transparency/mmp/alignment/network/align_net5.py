from typing import List, Iterable, Callable, Dict, Tuple, Set, Any
import tensorflow as tf
from keras.utils import losses_utils
from transformers import BertConfig, TFBertForSequenceClassification
from trainer_v2.per_project.transparency.mmp.alignment.network.align_net_v3 import build_align_acc_dict
from trainer_v2.per_project.transparency.mmp.trnsfmr_util import get_dummy_input_for_bert_layer
import math
from dataclasses import dataclass
from typing import List, Iterable, Callable, Dict, Tuple, Set, Any
import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils
from transformers import TFBertMainLayer, shape_list, BertConfig, TFBertForSequenceClassification
from transformers.tf_utils import stable_softmax

from trainer_v2.per_project.transparency.mmp.trnsfmr_util import get_dummy_input_for_bert_layer


class GAlignNetwork5:
    def __init__(self, tokenizer, classifier_layer_builder):
        bert_config = BertConfig()
        bert_cls = TFBertForSequenceClassification(bert_config)
        bert_main_layer = bert_cls.bert
        _ = bert_main_layer(get_dummy_input_for_bert_layer())
        cls_id = tokenizer.vocab["[CLS]"]
        sep_id = tokenizer.vocab["[SEP]"]

        # We skip dropout
        # Part 1. Build inputs
        max_term_len = 1
        q_term = tf.keras.layers.Input(shape=(max_term_len,), dtype='int32', name="q_term")
        d_term = tf.keras.layers.Input(shape=(max_term_len,), dtype='int32', name="d_term")
        raw_label = tf.keras.layers.Input(shape=(1,), dtype='float32', name="raw_label")
        label = tf.keras.layers.Input(shape=(1,), dtype='int32', name="label")
        is_valid = tf.keras.layers.Input(shape=(1,), dtype='int32', name="is_valid")
        inputs = [q_term, d_term, raw_label, label, is_valid]

        B, _ = shape_list(q_term)
        CLS = tf.ones([B, 1], tf.int32) * cls_id
        SEP = tf.ones([B, 1], tf.int32) * sep_id
        ZERO = tf.zeros([B, 1], tf.int32)
        input_ids = tf.concat([CLS, q_term, SEP, d_term, SEP], axis=1)
        seg1_len = max_term_len + 2
        seg2_len = max_term_len + 1

        token_type_ids_row = [0] * seg1_len + [1] * seg2_len
        token_type_ids = tf.tile(tf.expand_dims(token_type_ids_row, 0), [B, 1])

        bert_output = bert_main_layer(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            training=False,
            return_dict=True,
            output_hidden_states=True,
        )
        hidden = bert_output.hidden_states
        hidden = [tf.stop_gradient(h) for h in hidden]
        hidden_cls = [h[:, 0] for h in hidden]
        feature_rep = tf.concat(hidden_cls, axis=1)
        classifier = classifier_layer_builder()
        align_pred = classifier(feature_rep)
        align_probe = {
            'align_pred': align_pred
        }

        output_d = {
            "align_probe": align_probe,
            "raw_label": raw_label,
            "label": label,
            "is_valid": is_valid,
        }

        self.probe_model_output: Dict[str, Any] = output_d
        self.model = tf.keras.models.Model(
            inputs=inputs, outputs=self.probe_model_output)

    def get_align_metrics(self) -> Dict[str, tf.keras.metrics.Metric]:
        d = self.probe_model_output["align_probe"]
        output_d = build_align_acc_dict(d)
        return output_d
