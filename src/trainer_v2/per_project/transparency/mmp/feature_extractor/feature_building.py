import math
from dataclasses import dataclass
import tensorflow as tf
from keras.utils import losses_utils
from transformers import BertConfig, TFBertMainLayer
from transformers.models.bert.modeling_tf_bert import TFBertLayer
from transformers.tf_utils import stable_softmax, shape_list
from typing import List, Iterable, Callable, Dict, Tuple, Set, Any

from list_lib import dict_value_map
from models.transformer.bert_common_v2 import create_attention_mask_from_input_mask
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.tt_model.net_common import find_layer

# define a model which encode first layer
# Features that we will extract
# Q_i : Query of i-th head
# K_i: Key of i-th head
# V_i : Value oof i-th head

# Document token side
# Q'_i:
# K'_i:
# V'_i:

# Interaction in Query side

# A_i: weighted sum of V'_i
# A: A_i concatenated
# A0_i: A_i concatenated with 0 padding
# g(A): dense, layernorm applied to A
# g(A0_i), dense, layernorm applied to A0_i
# FF_out: Feed-forward layer output after residual combined with g(A)
# FF0_out: Feed-forward layer output after residual combined with g(A0_i)


# This layer corresponds to TFBertLayer
# TFBertLayer
# > TFBertAttention
# > > TFBertSelfAttention
# > > TFBertSelfOutput
# > TFBertIntermediate Begin
# > TFBertOutput Begin
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


def reduce_max(tensor, axis_arr):
    for axis in axis_arr:
        tensor = tf.reduce_max(tensor, axis)
    return tensor


def build_probs_from_tensor_d(d: Dict[str, tf.Tensor], out_dim=1) -> Dict[str, tf.Tensor]:
    probe_tensor = {}
    for k, feature_tensor in d.items():
        probe_name = "probe_on_" + k
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
#
def build_align_features(out_d, attn_mask, q_mask, d_mask) -> Dict[str, tf.Tensor]:
    # Features 1) Attention score between q_term/d_term
    # Features 2) Attention prob between q_term/d_term
    # Features 3) Global attention output
    # Features 4)
    mask = tf.cast(tf.expand_dims(attn_mask, axis=1), tf.float32)
    q_mask_ex = tf.cast(tf.expand_dims(q_mask, axis=2), tf.float32)
    d_mask_ex = tf.cast(tf.expand_dims(d_mask, axis=2), tf.float32)

    # [B, H]
    attn_scores_based = reduce_max(out_d['attention_scores'] * mask, [2, 2])
    attn_probs_based = reduce_max(out_d['attention_probs'] * mask, [2, 2])

    out_feature_set = {}
    out_feature_set['attn_scores'] = tf.expand_dims(attn_scores_based, axis=2)
    out_feature_set['attn_probs'] = tf.expand_dims(attn_probs_based, axis=2)

    # For features that has shape of [B, M, D]
    linear_features = [
        'attention_output',
        'g_attention_output',
        'g_attention_output_add_residual',
        'intermediate_output',
        'bert_out_last'
    ]

    all_linear_features_vectors = []
    for key in linear_features:
        # # [B, H]
        key_tm = key + "_tm"
        base_out = tf.reduce_sum(out_d[key] * q_mask_ex, axis=1)
        alt_out = tf.reduce_sum(out_d[key_tm] * q_mask_ex, axis=1)

        all_linear_features_vectors.append(base_out)
        all_linear_features_vectors.append(alt_out)

        pair_concat = tf.concat([base_out, alt_out], axis=1)
        out_feature_set[key + "paired"] = pair_concat

    out_feature_set['all_concat'] = tf.concat(all_linear_features_vectors, axis=1)
    return out_feature_set


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


def attention_mask_from_input_ids(input_ids):
    input_mask = tf.cast(input_ids == 0, tf.int32)
    attention_mask = create_attention_mask_from_input_mask(
        input_ids, input_mask)
    return attention_mask


def stop_gradient_for_dict(d: Dict[str, tf.Tensor]):
    return {k: tf.stop_gradient(v, name=f"{k}_stop_gradient") for k, v in d.items()}



def compute_logit_loss(target_logit, pred_logit):
    error = tf.abs(target_logit - pred_logit)
    tolerance = 0.5
    return tf.maximum(error, tolerance)


Metric = tf.keras.metrics.Metric
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
        # self.metric_inner.update_state(y_true, y_pred, sample_weight)
        mae = tf.reduce_sum(tf.abs(y_true_ex - y_pred))
        self.mae.assign_add(mae)
        n_valid = tf.reduce_sum(tf.cast(sample_weight, tf.float32))
        self.count.assign_add(n_valid)

    def result(self):
        # return self.metric_inner.result()
        return self.mae / self.count

    def reset_state(self):
        # self.metric_inner.reset_state()
        self.mae.assign(0.0)
        self.count.assign(0.0)


class ProbeLossFromDict(tf.keras.losses.Loss):
    def __init__(self, name='probe_loss_from_dict'):
        super().__init__(name=name)
        delta = 0.5
        self.base_loss_fn = tf.keras.losses.Huber(delta=delta, reduction=losses_utils.ReductionV2.NONE)

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, output_d):
        input_mask = output_d['input_mask']
        probe_on_hidden = output_d['probe_on_hidden']
        probe_on_attn_like = output_d['probe_on_attn_like']
        logits = output_d['logits']

        def loss_fn(target, pred):
            target_ex = tf.expand_dims(target, axis=1)
            losses = self.base_loss_fn(target_ex, pred, sample_weight=sample_weight)
            return tf.reduce_mean(losses)
        sample_weight = tf.cast(input_mask, tf.float32)

        loss_d = {}
        for k, v in probe_on_hidden.items():
            loss_d[f"probe/{k}_loss"] = loss_fn(logits, v)
        for k, v in probe_on_attn_like.items():
            loss_d[f"probe/{k}_loss"] = loss_fn(logits, v)

        loss_d_values = loss_d.values()
        loss = tf.reduce_sum(list(loss_d_values))
        return loss


# Probe predictor, Effect Predictor, G-Align predictor
# Evaluation metric: Term effect measure
class ProbeOnBERT:
    def __init__(self, bert_model):
        n_out_dim = 1
        target_layer_no = 0

        # We skip dropout
        bert_main_layer = find_layer(bert_model, "bert")
        orig_classifier = find_layer(bert_model, "classifier")
        bert_config = bert_main_layer._config
        c_log.info("identify_layers")
        layers_d = identify_layers(bert_main_layer, target_layer_no)

        input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
        segment_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="token_type_ids")
        target_q_term_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="target_q_term_mask")
        target_d_term_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="target_d_term_mask")
        inputs = [
            input_ids,
            segment_ids,
            target_q_term_mask,
            target_d_term_mask
        ]

        qd_target_mask = combine_qd_mask(target_q_term_mask, target_d_term_mask)

        c_log.info("bert_main_layer")
        bert_outputs = bert_main_layer(
            [input_ids, segment_ids],
            output_attentions=True,
            output_hidden_states=True)
        logits = orig_classifier(bert_outputs.pooler_output)
        logits = tf.stop_gradient(logits, "logits_stop_gradient")
        attn = bert_outputs.attentions
        hidden = bert_outputs.hidden_states
        hidden = [tf.stop_gradient(h) for h in hidden]

        input_mask = tf.cast(input_ids == 0, tf.int32)
        input_mask_shape = shape_list(input_mask)
        attn_mask = tf.reshape(
            input_mask, (input_mask_shape[0], 1, 1, input_mask_shape[1])
        )
        embedding = hidden[0]
        dtype = embedding.dtype
        attn_mask_bias = self.get_attn_mask_bias(attn_mask, dtype)

        c_log.info("TFBertLayerFlat")
        layer1_hidden = hidden[1]
        bert_flat = TFBertLayerFlat(bert_config, layers_d)
        per_layer_feature_tensors = bert_flat(layer1_hidden, attn_mask_bias, qd_target_mask)

        c_log.info("probe_on_hidden")
        hidden_tensor_d = {}
        for i, hidden_layer in enumerate(hidden):
            if 3 < i < 10:
                continue
            key = 'layer_{}'.format(i)
            hidden_tensor_d[key] = hidden_layer

        probe_on_hidden: Dict[str, tf.Tensor] = build_probs_from_tensor_d(hidden_tensor_d)

        c_log.info("probe_on_attn_like")
        probe_on_attn_like = build_probe_from_layer_features(
            per_layer_feature_tensors, bert_config.hidden_size, n_out_dim)
        #  dd
        c_log.info("build_align_features")
        align_feature_d = build_align_features(per_layer_feature_tensors,
                             qd_target_mask, target_q_term_mask, target_d_term_mask)
        align_probe: Dict[str, tf.Tensor] = {}
        self.probe_model_output: Dict[str, Any] = {
            "probe_on_hidden": probe_on_hidden,  # Dict[str, tf.Tensor]
            "probe_on_attn_like": probe_on_attn_like,
            "align_probe": align_probe,
            "logits": logits,
            "input_mask": input_mask,
        }
        self.model = tf.keras.models.Model(inputs=inputs, outputs=self.probe_model_output)

    def get_attn_mask_bias(self, attn_mask, dtype):
        one_cst = tf.constant(1.0, dtype=dtype)
        ten_thousand_cst = tf.constant(-10000.0, dtype=dtype)
        attn_mask = tf.cast(attn_mask, dtype=dtype)
        attn_mask_bias = tf.multiply(tf.subtract(one_cst, attn_mask), ten_thousand_cst)
        return attn_mask_bias

    def get_metrics(self) -> Dict[str, Metric]:
        output_d = {}
        target_key = 'logits'
        short_mapping = {
            "probe_on_hidden": "hidden",
            "probe_on_attn_like": "attn"
        }
        for probe_group in ["probe_on_hidden", "probe_on_attn_like"]:
            d = self.probe_model_output[probe_group]
            # with tf.name_scope(probe_group):
            for pred_key, out_tensor in d.items():
                group_short = short_mapping[probe_group]
                metric_name = f"{group_short}/{pred_key}"
                metric = ProbeMAE(target_key, probe_group, pred_key, name=metric_name)
                output_d[metric_name] = metric
        return output_d


def combine_qd_mask(target_q_term_mask, target_d_term_mask):
    a = tf.expand_dims(target_q_term_mask, axis=2)
    b = tf.expand_dims(target_d_term_mask, axis=1)
    qd_target_mask = a * b
    return qd_target_mask
