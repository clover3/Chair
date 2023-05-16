import math
from typing import Dict

import tensorflow as tf
from transformers import shape_list, TFBertMainLayer, BertConfig
from transformers.models.bert.modeling_tf_bert import TFBertLayer
from transformers.tf_utils import stable_softmax

from list_lib import dict_value_map


def combine_qd_mask(q_term_mask, d_term_mask):
    a = tf.expand_dims(q_term_mask, axis=2)
    b = tf.expand_dims(d_term_mask, axis=1)
    qd_target_mask = a * b
    return qd_target_mask


def build_paired_inputs_concat(keys):
    input_d = {}
    inputs = []
    for i in [1, 2]:
        for key in keys:
            field_name = key + str(i)
            input_tensor = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name=field_name)
            inputs.append(input_tensor)
            input_d[field_name] = input_tensor
    input_concat_d = {}
    for key in keys:
        t1 = input_d[key + "1"]
        t2 = input_d[key + "2"]
        t_concat = tf.concat([t1, t2], axis=0)
        input_concat_d[key] = t_concat
    return input_concat_d, inputs


def compute_probe_loss(logits, probe_d, base_loss_fn):
    def loss_fn(target, pred):
        losses = base_loss_fn(target, pred)
        return tf.reduce_mean(losses)

    loss_d = {}
    for k, v in probe_d.items():
        loss_d[f"probe/{k}_loss"] = loss_fn(logits, v)
    loss_d_values = loss_d.values()
    loss = tf.reduce_sum(list(loss_d_values))
    return loss


Metric = tf.keras.metrics.Metric

def build_probs_from_tensor_d(d: Dict[str, tf.Tensor], out_dim=1) -> Dict[str, tf.Tensor]:
    probe_tensor = {}
    for k, feature_tensor in d.items():
        probe_name = k
        dense = tf.keras.layers.Dense(out_dim, name=probe_name)
        probe_tensor[probe_name] = dense(feature_tensor)
    return probe_tensor


def build_probe_from_layer_features(out_d, all_head_size, out_dim):
    bhmd_features = ['query_layer', 'key_layer', 'value_layer']
    bhmm_features = ['attention_scores', 'attention_probs',
                     'attention_scores_tm', 'attention_probs_tm',]
    bmd_features = ['attention_output', 'g_attention_output', 'g_attention_output_add_residual',
                    'intermediate_output_tm', 'bert_out_last_tm',
                    'attention_output_tm', 'g_attention_output_tm', 'g_attention_output_add_residual_tm',
                    'intermediate_output_tm', 'bert_out_last_tm',
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

    prediction_out_d = {}
    prediction_out_d.update(bhmd_probe_d)
    prediction_out_d.update(bmd_probe_d)
    return prediction_out_d


    # The goal of this classifier is to predict if given q tokens and d tokens are considered globally aligned


def identify_layers(bert_main_layer: TFBertMainLayer, target_layer_no):
    layer: TFBertLayer = bert_main_layer.encoder.layer[target_layer_no]
    layers_d = identify_layers_from_bert_layer(layer)
    return layers_d


def identify_layers_from_bert_layer(layer: TFBertLayer):
    layers_d = {}
    self_attention = layer.attention.self_attention
    layers_d['query'] = self_attention.query
    layers_d['key'] = self_attention.key
    layers_d['value'] = self_attention.value
    layers_d['attn_out_dense'] = layer.attention.dense_output.dense
    layers_d['attn_out_layernorm'] = layer.attention.dense_output.LayerNorm
    layers_d['intermediate_dense'] = layer.intermediate.dense
    layers_d['intermediate_act_fn'] = layer.intermediate.intermediate_act_fn
    layers_d['bert_out_dense'] = layer.bert_output.dense
    layers_d['bert_out_layernorm'] = layer.bert_output.LayerNorm
    return layers_d


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
        target_mask: tf.Tensor,
    ):
        target_mask_ex = tf.cast(tf.expand_dims(target_mask, axis=1), tf.float32)

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

        attention_probs_tm = attention_probs * target_mask_ex
        attention_output_tm = tf.matmul(attention_probs_tm, value_layer)
        attention_output_tm = tf.transpose(attention_output_tm, perm=[0, 2, 1, 3])
        attention_output_tm = tf.reshape(tensor=attention_output_tm, shape=(batch_size, -1, self.all_head_size))

        #  > > TFBertSelfAttention End

        #  > > TFBertSelfOutput Begin
        # (batch_size, seq_len_q, all_head_size)
        g_attention_output = self.attn_out_dense(inputs=attention_output)
        g_attention_output_tm = self.attn_out_dense(inputs=attention_output_tm)

        g_attention_output_add_residual = self.attn_out_layernorm(inputs=g_attention_output + layer_input_vector)
        g_attention_output_add_residual_tm = self.attn_out_layernorm(inputs=g_attention_output_tm + layer_input_vector)

        #  > > TFBertSelfOutput End
        # > TFBertAttention End

        # > TFBertIntermediate Begin
        hidden_states = self.intermediate_dense(inputs=g_attention_output_add_residual)
        hidden_states_tm = self.intermediate_dense(inputs=g_attention_output_add_residual_tm)
        intermediate_output = self.intermediate_act_fn(hidden_states)
        intermediate_output_tm = self.intermediate_act_fn(hidden_states_tm)

        # > TFBertOutput Begin
        bert_out_first = self.bert_out_dense(inputs=intermediate_output)
        bert_out_first_tm = self.bert_out_dense(inputs=intermediate_output_tm)

        bert_out_last = self.bert_out_layernorm(inputs=bert_out_first + g_attention_output_add_residual)
        bert_out_last_tm = self.bert_out_layernorm(inputs=bert_out_first_tm + g_attention_output_add_residual_tm)
        layer_output = bert_out_last
        out_d = {
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

            'attention_probs_tm': attention_probs_tm,
            'attention_output_tm': attention_output_tm,
            'g_attention_output_tm': g_attention_output_tm,
            'g_attention_output_add_residual_tm': g_attention_output_add_residual_tm,
            'intermediate_output_tm': intermediate_output_tm,
            'bert_out_last_tm': bert_out_last_tm
        }

        out_d = stop_gradient_for_dict(out_d)
        return out_d


def stop_gradient_for_dict(d: Dict[str, tf.Tensor]):
    return {k: tf.stop_gradient(v, name=f"{k}_stop_gradient") for k, v in d.items()}




class ProbeMAE(Metric):
    def __init__(self, target_key, pred_parent, pred_key, name, **kwargs):
        super(ProbeMAE, self).__init__(name=name, **kwargs)
        self.mae = self.add_weight(name='mae', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        # self.metric_inner = tf.keras.metrics.MeanAbsoluteError()
        self.target_key = target_key
        self.pred_parent = pred_parent
        self.pred_key = pred_key

    def update_state(self, output_d, _sample_weight=None):
        y_true = output_d[self.target_key]
        y_true_ex = tf.expand_dims(y_true, 1)
        y_pred = output_d[self.pred_parent][self.pred_key]
        input_mask = output_d['input_mask']
        sample_weight = tf.cast(input_mask, tf.float32)
        mae = tf.reduce_sum(tf.abs(y_true_ex - y_pred))
        self.mae.assign_add(mae)
        n_valid = tf.reduce_sum(tf.cast(sample_weight, tf.float32))
        self.count.assign_add(n_valid)

    def result(self):
        return self.mae / self.count

    def reset_state(self):
        self.mae.assign(0.0)
        self.count.assign(0.0)


class ProbePairwiseAcc(Metric):
    def __init__(self, target_key, pred_parent, pred_key, name, **kwargs):
        super(ProbePairwiseAcc, self).__init__(name=name, **kwargs)
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        self.target_key = target_key
        self.pred_parent = pred_parent
        self.pred_key = pred_key

    def update_state(self, output_d, _sample_weight=None):
        y_pred = output_d[self.pred_parent][self.pred_key]  # [B*2, M, 1] or  [B, 1]
        input_mask = output_d['input_mask']
        shape_l = shape_list(y_pred)

        if len(shape_l) == 3:
            B2, M, One = shape_l
            y_pred_paired = tf.reshape(y_pred, [2, -1, M, 1])
            input_mask_paired = tf.reshape(input_mask, [2, -1, M, 1])
        elif len(shape_l) == 2:
            B2, One = shape_l
            y_pred_paired = tf.reshape(y_pred, [2, -1, 1])
            input_mask_paired = tf.ones_like(y_pred_paired, tf.int32)
        else:
            raise Exception()


        y_pred_pos = y_pred_paired[0]
        y_pred_neg = y_pred_paired[1]

        input_mask_pos = input_mask_paired[0]
        input_mask_neg = input_mask_paired[1]

        input_mask_both = input_mask_pos * input_mask_neg
        sample_weight = tf.cast(input_mask_both, tf.float32)
        sample_weight_f = tf.cast(sample_weight, tf.float32)

        is_correct = tf.cast(y_pred_pos > y_pred_neg, tf.float32)
        correct_masked_f = tf.reduce_sum(is_correct * sample_weight_f)

        self.correct.assign_add(correct_masked_f)
        n_valid = tf.reduce_sum(sample_weight_f)
        self.count.assign_add(n_valid)

    def result(self):
        return self.correct / self.count

    def reset_state(self):
        self.correct.assign(0.0)
        self.count.assign(0.0)

