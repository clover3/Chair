import tensorflow as tf
from keras.utils import losses_utils
from transformers import TFBertMainLayer, PretrainedConfig, BertConfig
from transformers.tf_utils import shape_list
from typing import Dict, Any

from cpath import get_bert_config_path
from models.transformer.bert_common_v2 import create_attention_mask_from_input_mask, get_shape_list
from trainer_v2.chair_logging import c_log
from tf_util.lib.tf_funcs import find_layer


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
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config
from trainer_v2.per_project.transparency.mmp.probe.probe_common import build_paired_inputs_concat, compute_probe_loss, \
    Metric, ProbeMAE, ProbePairwiseAcc, build_probs_from_tensor_d, build_probe_from_layer_features, identify_layers, \
    TFBertLayerFlat
from trainer_v2.per_project.transparency.mmp.trnsfmr_util import get_dummy_input_for_bert_layer


def attention_mask_from_input_ids(input_ids):
    input_mask = tf.cast(tf.equal(input_ids, 0), tf.int32)
    attention_mask = create_attention_mask_from_input_mask(
        input_ids, input_mask)
    return attention_mask


def compute_logit_loss(target_logit, pred_logit):
    error = tf.abs(target_logit - pred_logit)
    tolerance = 0.5
    return tf.maximum(error, tolerance)


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
        sample_weight = tf.cast(input_mask, tf.float32)
        def loss_fn(target, pred):
            target_ex = tf.expand_dims(target, axis=1)
            return self.base_loss_fn(target_ex, pred, sample_weight=sample_weight)

        all_d = {}
        all_d.update(probe_on_hidden)
        all_d.update(probe_on_attn_like)
        return compute_probe_loss(logits, all_d, loss_fn)



class ProbePairwise(tf.keras.losses.Loss):
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
        logits = output_d['logits'] # [B*2, 1]

        logits_pair = tf.reshape(logits, [2, -1, 1, 1])
        targets = logits_pair[0] - logits_pair[1]
        pair_label = tf.cast(tf.less(0, targets), tf.int32)
        B, _, _= get_shape_list(targets)

        sample_weight = tf.cast(input_mask, tf.float32)
        margin = 0.5
        f_pair_gap_large = tf.cast(tf.less(margin, tf.abs(targets)), tf.float32)
        sample_weight = sample_weight * f_pair_gap_large

        def loss_per_probe(pred):
            pred_pair = tf.reshape(pred, [2, B, -1, 1])
            pred_v = pred_pair[0] - pred_pair[1]
            losses =  self.base_loss_fn(pair_label, pred_v, sample_weight=sample_weight)
            return tf.reduce_mean(losses)

        all_probe_d = {}
        all_probe_d.update(probe_on_hidden)
        all_probe_d.update(probe_on_attn_like)

        loss_d = {}
        for k, v in all_probe_d.items():
            loss_d[f"probe/{k}_loss"] = loss_per_probe(logits, v)
        loss_d_values = loss_d.values()
        loss = tf.reduce_sum(list(loss_d_values))
        return loss


# Probe predictor, Effect Predictor, G-Align predictor
# Evaluation metric: Term effect measure
def build_bert_layer(old_bert_layer):
    new_bert_layer = TFBertMainLayer(old_bert_layer._config, name="bert")
    param_values = tf.keras.backend.batch_get_value(old_bert_layer.weights)
    _ = new_bert_layer(get_dummy_input_for_bert_layer())
    tf.keras.backend.batch_set_value(zip(new_bert_layer.weights, param_values))
    return new_bert_layer


class ProbeOnBERT:
    def __init__(self, bert_model):
        n_out_dim = 1
        target_layer_no = 0
        # We skip dropout
        bert_main_layer_ckpt = find_layer(bert_model, "bert")
        classifier_ckpt = find_layer(bert_model, "classifier")

        classifier = classifier_ckpt
        bert_main_layer = bert_main_layer_ckpt
        bert_config = bert_main_layer_ckpt._config

        c_log.info("identify_layers")
        layers_d = identify_layers(bert_main_layer, target_layer_no)

        keys = ["input_ids", "token_type_ids"]
        input_concat_d, inputs = build_paired_inputs_concat(keys)

        input_ids = input_concat_d["input_ids"]
        segment_ids = input_concat_d["token_type_ids"]
        bert_input = {
            'input_ids': input_ids,
            'token_type_ids': segment_ids
        }
        c_log.info("bert_main_layer")
        bert_outputs = bert_main_layer(
            bert_input,
            output_attentions=True,
            output_hidden_states=True)
        logits = classifier(bert_outputs.pooler_output)
        logits = tf.stop_gradient(logits, "logits_stop_gradient")
        hidden = bert_outputs.hidden_states
        hidden = [tf.stop_gradient(h) for h in hidden]

        input_mask = tf.cast(tf.not_equal(input_ids, 0), tf.int32)
        input_mask_shape = shape_list(input_mask)
        B = input_mask_shape[0]
        M = input_mask_shape[1]
        attn_mask = tf.reshape(
            input_mask, (input_mask_shape[0], 1, 1, input_mask_shape[1])
        )
        embedding = hidden[0]
        dtype = embedding.dtype
        attn_mask_bias = self.get_attn_mask_bias(attn_mask, dtype)
        c_log.info("TFBertLayerFlat")
        layer1_hidden = hidden[1]
        qd_target_mask = tf.zeros([B, M, M], tf.int32)
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
        self.probe_model_output: Dict[str, Any] = {
            "probe_on_hidden": probe_on_hidden,  # Dict[str, tf.Tensor]
            "probe_on_attn_like": probe_on_attn_like,
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

    def get_probe_metrics(self) -> Dict[str, Metric]:
        output_d = {}
        target_key = 'logits'
        short_mapping = {
            "probe_on_hidden": "hidden",
            "probe_on_attn_like": "attn"
        }
        for probe_group in ["probe_on_hidden", "probe_on_attn_like"]:
            d = self.probe_model_output[probe_group]
            for pred_key, out_tensor in d.items():
                group_short = short_mapping[probe_group]
                metric_name = f"{group_short}/{pred_key}"
                metric = ProbeMAE(target_key, probe_group, pred_key, name=metric_name)
                output_d[metric_name] = metric

                metric_name = f"{group_short}/{pred_key}/PairAcc"
                metric = ProbePairwiseAcc(target_key, probe_group, pred_key, name=metric_name)
                output_d[metric_name] = metric

        return output_d


class ProbeAndAttention:
    def __init__(self, bert_model):
        n_out_dim = 1
        target_layer_no = 0
        # We skip dropout
        bert_main_layer_ckpt = find_layer(bert_model, "bert")
        _ = bert_main_layer_ckpt(get_dummy_input_for_bert_layer())

        bert_main_layer = bert_main_layer_ckpt
        bert_config = bert_main_layer_ckpt._config

        c_log.info("identify_layers")
        layers_d = identify_layers(bert_main_layer, target_layer_no)

        keys = ["input_ids", "token_type_ids"]
        input_concat_d, inputs = build_paired_inputs_concat(keys)

        input_ids = input_concat_d["input_ids"]
        segment_ids = input_concat_d["token_type_ids"]
        bert_input = {
            'input_ids': input_ids,
            'token_type_ids': segment_ids
        }
        c_log.info("bert_main_layer")
        bert_outputs = bert_main_layer(
            bert_input,
            output_attentions=True,
            output_hidden_states=True)
        hidden = bert_outputs.hidden_states
        hidden = [tf.stop_gradient(h) for h in hidden]
        attentions = bert_outputs.attentions

        input_mask = tf.cast(tf.not_equal(input_ids, 0), tf.int32)
        input_mask_shape = shape_list(input_mask)
        B = input_mask_shape[0]
        M = input_mask_shape[1]
        attn_mask = tf.reshape(
            input_mask, (input_mask_shape[0], 1, 1, input_mask_shape[1])
        )
        embedding = hidden[0]
        dtype = embedding.dtype
        attn_mask_bias = self.get_attn_mask_bias(attn_mask, dtype)
        c_log.info("TFBertLayerFlat")
        layer1_hidden = hidden[1]
        qd_target_mask = tf.zeros([B, M, M], tf.int32)
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
        self.probe_model_output: Dict[str, Any] = {
            "probe_on_hidden": probe_on_hidden,  # Dict[str, tf.Tensor]
            "probe_on_attn_like": probe_on_attn_like,
            "input_ids": input_ids,
            "input_mask": input_mask,
            "attentions": attentions
        }
        self.model = tf.keras.models.Model(inputs=inputs, outputs=self.probe_model_output)

    def get_attn_mask_bias(self, attn_mask, dtype):
        one_cst = tf.constant(1.0, dtype=dtype)
        ten_thousand_cst = tf.constant(-10000.0, dtype=dtype)
        attn_mask = tf.cast(attn_mask, dtype=dtype)
        attn_mask_bias = tf.multiply(tf.subtract(one_cst, attn_mask), ten_thousand_cst)
        return attn_mask_bias

    def get_probe_metrics(self) -> Dict[str, Metric]:
        output_d = {}
        target_key = 'logits'
        short_mapping = {
            "probe_on_hidden": "hidden",
            "probe_on_attn_like": "attn"
        }
        for probe_group in ["probe_on_hidden", "probe_on_attn_like"]:
            d = self.probe_model_output[probe_group]
            for pred_key, out_tensor in d.items():
                group_short = short_mapping[probe_group]
                metric_name = f"{group_short}/{pred_key}"
                metric = ProbeMAE(target_key, probe_group, pred_key, name=metric_name)
                output_d[metric_name] = metric

                metric_name = f"{group_short}/{pred_key}/PairAcc"
                metric = ProbePairwiseAcc(target_key, probe_group, pred_key, name=metric_name)
                output_d[metric_name] = metric

        return output_d