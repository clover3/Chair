from dataclasses import dataclass
from typing import Dict, Any
import tensorflow as tf
from transformers import shape_list, BertConfig, TFBertForSequenceClassification

from trainer_v2.per_project.transparency.mmp.alignment.network.align_net_v2 import TFBertLayerFlat
from trainer_v2.per_project.transparency.mmp.alignment.network.common import reshape_per_head_features, \
    mean_pool_over_masked, get_emb_concat_feature, build_align_acc_dict
from trainer_v2.per_project.transparency.mmp.probe.probe_common import identify_layers, \
    combine_qd_mask, get_attn_mask_bias
from trainer_v2.per_project.transparency.mmp.trnsfmr_util import get_dummy_input_for_bert_layer


@dataclass
class ThresholdConfig:
    threshold_upper: float
    threshold_lower: float


def build_probe_from_layer_features(
        out_d: Dict[str, tf.Tensor], hidden_size, projection_layer, seq_pooling_fn):
    bmd_hidden_vars = [
        'layer_input_vector', 'attention_output',
        'g_attention_output', 'g_attention_output_add_residual',
        'intermediate_output', 'bert_out_last']
    bhmd_hidden_var_d = reshape_per_head_features(out_d, hidden_size)
    bmd_hidden_var_d = {k: out_d[k] for k in bmd_hidden_vars}

    hidden_vars = []
    hidden_vars.extend(bhmd_hidden_var_d.values())
    hidden_vars.extend(bmd_hidden_var_d.values())
    all_feature = tf.concat(hidden_vars, axis=2)

    arr = []
    arr.extend(bhmd_hidden_var_d.values())
    qkv_feature = tf.concat(arr, axis=2)

    concat_hidden_var_d = {
        "all_concat": all_feature,
        "qkv_feature": qkv_feature,
    }

    all_hidden_var_d = {}
    all_hidden_var_d.update(bhmd_hidden_var_d)
    all_hidden_var_d.update(bmd_hidden_var_d)
    all_hidden_var_d.update(concat_hidden_var_d)

    probe_tensor_d = {}
    for k, feature_tensor in all_hidden_var_d.items():
        probe_name = k
        pooled = seq_pooling_fn(feature_tensor)
        projection = projection_layer(probe_name)
        probe_tensor_d[probe_name] = projection(pooled)

    return probe_tensor_d


class GAlignNetwork3:
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

        def pool_by_q_term_mask(vector):
            return mean_pool_over_masked(vector, q_term_mask)

        seg1_len = max_term_len + 2
        seg2_len = max_term_len + 1

        token_type_ids_row = [0] * seg1_len + [1] * seg2_len
        token_type_ids = tf.tile(tf.expand_dims(token_type_ids_row, 0), [B, 1])

        embedding_output = bert_main_layer.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            training=False,
        )
        qd_target_mask = combine_qd_mask(q_term_mask, d_term_mask)
        input_mask = tf.cast(tf.not_equal(input_ids, 0), tf.int32)
        input_mask_shape = shape_list(input_mask)
        attn_mask = tf.reshape(
            input_mask, (input_mask_shape[0], 1, 1, input_mask_shape[1])
        )
        dtype = embedding_output.dtype
        attn_mask_bias = get_attn_mask_bias(attn_mask, dtype)

        bert_flat = TFBertLayerFlat(bert_config, layers_d)
        per_layer_feature_tensors = bert_flat(embedding_output, attn_mask_bias)

        def projection_layer(probe_name):
            return tf.keras.layers.Dense(n_out_dim, name=probe_name)

        align_probe = build_probe_from_layer_features(
            per_layer_feature_tensors, bert_config.hidden_size, projection_layer, pool_by_q_term_mask)

        emb_concat_probe = get_emb_concat_feature(bert_main_layer, projection_layer, q_term, d_term)
        align_probe['emb_concat'] = emb_concat_probe
        for k, v in align_probe.items():
            print(k, v)

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


