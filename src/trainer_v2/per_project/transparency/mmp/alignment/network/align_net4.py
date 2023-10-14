from typing import List, Iterable, Callable, Dict, Tuple, Set, Any
import tensorflow as tf
from keras.utils import losses_utils
from transformers import BertConfig, TFBertForSequenceClassification
from trainer_v2.per_project.transparency.mmp.alignment.network.common import build_align_acc_dict
from trainer_v2.per_project.transparency.mmp.trnsfmr_util import get_dummy_input_for_bert_layer


class GAlignNetwork4:
    def __init__(self, classifier_layer_builder):
        bert_config = BertConfig()
        bert_cls = TFBertForSequenceClassification(bert_config)
        bert_main_layer = bert_cls.bert
        _ = bert_main_layer(get_dummy_input_for_bert_layer())

        max_term_len = 1
        q_term = tf.keras.layers.Input(shape=(max_term_len,), dtype='int32', name="q_term")
        d_term = tf.keras.layers.Input(shape=(max_term_len,), dtype='int32', name="d_term")
        raw_label = tf.keras.layers.Input(shape=(1,), dtype='float32', name="raw_label")
        label = tf.keras.layers.Input(shape=(1,), dtype='int32', name="label")
        is_valid = tf.keras.layers.Input(shape=(1,), dtype='int32', name="is_valid")
        inputs = [q_term, d_term, raw_label, label, is_valid]

        q_term_emb = bert_main_layer.embeddings(input_ids=q_term)
        d_term_emb = bert_main_layer.embeddings(input_ids=d_term)
        t = tf.concat([q_term_emb, d_term_emb], axis=2)[:, 0, :]
        k = "emb_concat"
        t_stop = tf.stop_gradient(t, name=f"{k}_stop_gradient")
        classifier = classifier_layer_builder()
        emb_concat_probe = classifier(t_stop)
        align_probe = {
            k: emb_concat_probe
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
