import math
from dataclasses import dataclass
from typing import List, Iterable, Callable, Dict, Tuple, Set, Any
import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils
from transformers import TFBertMainLayer, shape_list, BertConfig, TFBertForSequenceClassification
from transformers.tf_utils import stable_softmax

from list_lib import dict_value_map
from trainer_v2.per_project.transparency.mmp.probe.probe_common import identify_layers, \
    stop_gradient_for_dict, build_probs_from_tensor_d, combine_qd_mask, get_attn_mask_bias
from trainer_v2.per_project.transparency.mmp.trnsfmr_util import get_dummy_input_for_bert_layer


@dataclass
class ThresholdConfig:
    threshold_upper: float
    threshold_lower: float


def build_probe_from_layer_features(out_d, all_head_size, out_dim):
    bhmd_features = ['query_layer', 'key_layer', 'value_layer']
    bmd_features = [
        # 'layer_input_vector',
        'attention_output',
        'g_attention_output', 'g_attention_output_add_residual',
        # 'intermediate_output', 'bert_out_last'
    ]

    batch_size = shape_list(out_d['query_layer'])[0]
    bhmd_feature_d = {k: out_d[k] for k in bhmd_features}
    def reshape_bhmd(tensor):
        t = tf.transpose(tensor, perm=[0, 2, 1, 3])
        t = tf.reshape(tensor=t, shape=(batch_size, -1, all_head_size))
        return t

    bhmd_feature_d = dict_value_map(reshape_bhmd, bhmd_feature_d)
    bhmd_probe_d = build_probs_from_tensor_d(bhmd_feature_d, out_dim)

    bmd_feature_d = {k: out_d[k] for k in bmd_features}
    bmd_probe_d = build_probs_from_tensor_d(bmd_feature_d, out_dim)

    all_features = []
    all_features.extend(bhmd_feature_d.values())
    all_features.extend(bmd_feature_d.values())
    all_feature = tf.concat(all_features, axis=2)
    arr = []
    arr.extend(bhmd_feature_d.values())
    qkv_feature = tf.concat(arr, axis=2)

    combined_feature_d = {
        "all_concat": all_feature,
        "qkv_feature": qkv_feature,
    }
    combined_probe = build_probs_from_tensor_d(combined_feature_d, out_dim)

    prediction_out_d = {}
    prediction_out_d.update(bhmd_probe_d)
    prediction_out_d.update(bmd_probe_d)
    prediction_out_d.update(combined_probe)
    return prediction_out_d


class TFBertLayerFlat(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, layers_d):
        super(TFBertLayerFlat, self).__init__()

        self.query = layers_d['query']
        self.key = layers_d['key']
        self.value = layers_d['value']
        self.attn_out_dense = layers_d['attn_out_dense']
        self.attn_out_layernorm = layers_d['attn_out_layernorm']

        self.intermediate_dense = layers_d['intermediate_dense']
        self.intermediate_act_fn = layers_d['intermediate_act_fn']

        self.bert_out_dense = layers_d['bert_out_dense']
        self.bert_out_layernorm = layers_d['bert_out_layernorm']

        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.num_attention_heads = config.num_attention_heads
        self.all_head_size = config.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # Reshape from [batch_size, seq_length, all_head_size] to [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))
        # Transpose the tensor from [batch_size, seq_length, num_attention_heads, attention_head_size] to [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])

    def call(
        self,
        layer_input_vector: tf.Tensor,
        attn_mask_bias: tf.Tensor,
    ):
        batch_size = shape_list(layer_input_vector)[0]
        # H: number of head
        # D: dimension per head
        # hidden_states: [B, M, H*D]

        #  > TFBertAttention Begin
        #  > > TFBertSelfAttention Begin
        mixed_query_layer = self.query(inputs=layer_input_vector)
        # [B, H, M, D]
        key_layer = self.transpose_for_scores(self.key(inputs=layer_input_vector), batch_size)
        value_layer = self.transpose_for_scores(self.value(inputs=layer_input_vector), batch_size)
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)

        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True) # [B, H, M, M]
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)
        attention_scores = tf.divide(attention_scores, dk)
        attention_scores = tf.add(attention_scores, attn_mask_bias)

        attention_probs = stable_softmax(logits=attention_scores, axis=-1)  # [B, H, M, M]
        attention_output = tf.matmul(attention_probs, value_layer)
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(tensor=attention_output, shape=(batch_size, -1, self.all_head_size))

        #  > > TFBertSelfAttention End

        #  > > TFBertSelfOutput Begin
        # (batch_size, seq_len_q, all_head_size)
        g_attention_output = self.attn_out_dense(inputs=attention_output)
        g_attention_output_add_residual = self.attn_out_layernorm(inputs=g_attention_output + layer_input_vector)
        #  > > TFBertSelfOutput End
        # > TFBertAttention End

        # > TFBertIntermediate Begin
        hidden_states = self.intermediate_dense(inputs=g_attention_output_add_residual)
        intermediate_output = self.intermediate_act_fn(hidden_states)

        # > TFBertOutput Begin
        bert_out_first = self.bert_out_dense(inputs=intermediate_output)

        bert_out_last = self.bert_out_layernorm(inputs=bert_out_first + g_attention_output_add_residual)
        layer_output = bert_out_last
        out_d = {
            'layer_input_vector': layer_input_vector,
            'query_layer': query_layer,
            'key_layer': key_layer,
            'value_layer': value_layer,
            'attention_scores': attention_scores,
            'attention_probs': attention_probs,
            'attention_output': attention_output,
            'g_attention_output': g_attention_output,
            'g_attention_output_add_residual': g_attention_output_add_residual,
            'intermediate_output': intermediate_output,
            'bert_out_last': bert_out_last,
        }
        out_d = stop_gradient_for_dict(out_d)
        return out_d


class GAlignNetworkSingleTerm:
    def __init__(self, tokenizer):
        n_out_dim = 1
        target_layer_no = 0
        cls_id = tokenizer.vocab["[CLS]"]
        sep_id = tokenizer.vocab["[SEP]"]

        bert_config = BertConfig()
        bert_cls = TFBertForSequenceClassification(bert_config)
        bert_main_layer = bert_cls.bert
        # bert_main_layer = TFBertMainLayer(bert_config)
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
        align_probe = build_probe_from_layer_features(
            per_layer_feature_tensors, bert_config.hidden_size, n_out_dim)
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
        output_d = {}
        d = self.probe_model_output["align_probe"]
        for pred_key, out_tensor in d.items():
            metric_name = f"align/{pred_key}"
            metric = AlignAcc("align_probe", pred_key, name=metric_name)
            output_d[metric_name] = metric
        return output_d


class AlignAcc(tf.keras.metrics.Metric):
    def __init__(self, pred_parent, pred_key, name, **kwargs):
        super(AlignAcc, self).__init__(name=name, **kwargs)
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        self.pred_parent = pred_parent
        self.pred_key = pred_key

    def update_state(self, output_d, _sample_weight=None):
        y_pred_score = output_d[self.pred_parent][self.pred_key] # [B, M, 1]
        label = output_d["label"]  # [B, 1]
        is_valid = output_d["is_valid"]  # [B, 1]
        sample_weight_per_label = tf.cast(is_valid, tf.float32) # [B, 1]
        q_term_mask = tf.cast(output_d["q_term_mask"], tf.float32) # [B, M]

        y_pred = y_pred_score > 0  # [B, M, 1]
        label_b = tf.expand_dims(tf.cast(label, tf.bool), axis=1) # [B, 1]
        is_correct = tf.cast(tf.equal(label_b, y_pred), tf.float32) # [B, M, 1]
        sample_weight = tf.expand_dims(sample_weight_per_label, axis=1) * tf.expand_dims(q_term_mask, axis=2)

        n_valid_correct = tf.reduce_sum(is_correct * sample_weight)  # scalar
        self.correct.assign_add(n_valid_correct)
        n_valid = tf.reduce_sum(sample_weight)
        self.count.assign_add(n_valid)

    def result(self):
        return self.correct / self.count

    def reset_state(self):
        self.correct.assign(0.0)
        self.count.assign(0.0)


class AlignLossFromDict(tf.keras.losses.Loss):
    def __init__(self, seq_len, name='align_loss_from_dict'):
        super().__init__(name=name)
        self.seq_len = seq_len
        self.base_loss_fn = tf.keras.losses.BinaryCrossentropy(
            from_logits=True,
            reduction=losses_utils.ReductionV2.NONE)

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, output_d):
        align_probe = output_d['align_probe'] # Each value has [B, M, 1]
        label = output_d['label']  # [B, 1]
        is_valid = output_d['is_valid']  # [B, 1]
        sample_weight_per_label = tf.cast(is_valid, tf.float32) # [B, 1]
        q_term_mask = tf.cast(output_d["q_term_mask"], tf.float32) # [B, M]
        sample_weight = tf.expand_dims(sample_weight_per_label, axis=1)\
                        * tf.expand_dims(q_term_mask, axis=2)

        loss_d = {}
        for k, pred in align_probe.items():
            label_ex = tf.expand_dims(label, axis=1)
            label_ex = tf.tile(label_ex, [1, self.seq_len, 1])
            losses = self.base_loss_fn(label_ex, pred, sample_weight=sample_weight)
            loss_d[f"align/{k}_loss"] = tf.reduce_mean(losses)

        loss_d_values = loss_d.values()
        loss = tf.reduce_sum(list(loss_d_values))
        return loss
