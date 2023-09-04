from trainer_v2.per_project.transparency.mmp.alignment.network.align_net_v2 import TFBertLayerFlat
from trainer_v2.per_project.transparency.mmp.alignment.network.align_net_v3 import build_align_acc_dict, \
    get_emb_concat_feature, mean_pool_over_masked, build_probe_from_layer_features
from dataclasses import dataclass
from typing import List, Iterable, Callable, Dict, Tuple, Set, Any
import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils
from transformers import TFBertMainLayer, shape_list, BertConfig, TFBertForSequenceClassification

from trainer_v2.per_project.transparency.mmp.probe.probe_common import get_attn_mask_bias, identify_layers
from trainer_v2.per_project.transparency.mmp.trnsfmr_util import get_dummy_input_for_bert_layer


class TwoLayerDense(tf.keras.layers.Layer):
    def __init__(self, hidden_size, hidden_size2,
                 activation1='relu',
                 **kwargs
                 ):
        super(TwoLayerDense, self).__init__(**kwargs)
        self.layer1 = tf.keras.layers.Dense(hidden_size, activation=activation1)
        self.layer2 = tf.keras.layers.Dense(hidden_size2)

    def call(self, inputs, *args, **kwargs):
        hidden = self.layer1(inputs)
        return self.layer2(hidden)


class GAlignNetwork6:
    def __init__(self, tokenizer):
        n_out_dim = 1
        target_layer_no = 0
        cls_id = tokenizer.vocab["[CLS]"]
        sep_id = tokenizer.vocab["[SEP]"]

        bert_config = BertConfig()
        bert_cls = TFBertForSequenceClassification(bert_config)
        bert_main_layer = bert_cls.bert
        _ = bert_main_layer(get_dummy_input_for_bert_layer())

        # We skip dropout
        layers_d = identify_layers(bert_main_layer, target_layer_no)

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
        q_term_mask = tf.concat([ZERO, tf.ones_like(q_term, tf.int32), ZERO,
                                 tf.zeros_like(d_term, tf.int32), ZERO], axis=1)
        d_term_mask = tf.concat([ZERO, tf.zeros_like(q_term, tf.int32), ZERO,
                                 tf.ones_like(d_term, tf.int32), ZERO], axis=1)

        seg1_len = max_term_len + 2
        seg2_len = max_term_len + 1

        token_type_ids_row = [0] * seg1_len + [1] * seg2_len
        token_type_ids = tf.tile(tf.expand_dims(token_type_ids_row, 0), [B, 1])

        embedding_output = bert_main_layer.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            training=False,
        )
        input_mask = tf.cast(tf.not_equal(input_ids, 0), tf.int32)
        input_mask_shape = shape_list(input_mask)
        attn_mask = tf.reshape(
            input_mask, (input_mask_shape[0], 1, 1, input_mask_shape[1])
        )
        dtype = embedding_output.dtype
        attn_mask_bias = get_attn_mask_bias(attn_mask, dtype)
        def pool_by_q_term_mask(vector):
            return mean_pool_over_masked(vector, q_term_mask)

        bert_flat = TFBertLayerFlat(bert_config, layers_d)
        per_layer_feature_tensors = bert_flat(embedding_output, attn_mask_bias)

        def projection_layer(probe_name):
            hidden_size = bert_config.hidden_size
            return TwoLayerDense(hidden_size, n_out_dim, name=probe_name)

        align_probe = build_probe_from_layer_features(
            per_layer_feature_tensors, bert_config.hidden_size, projection_layer, pool_by_q_term_mask)

        emb_concat_probe = get_emb_concat_feature(bert_main_layer, projection_layer, q_term, d_term)
        align_probe['emb_concat'] = emb_concat_probe
        for k, v in align_probe.items():
            print(k, v)

        align_probe['align_pred'] = align_probe['g_attention_output']
        output_d = {
            "align_probe": align_probe,
            "input_mask": input_mask,
            "q_term_mask": q_term_mask,
            "d_term_mask": d_term_mask,
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
