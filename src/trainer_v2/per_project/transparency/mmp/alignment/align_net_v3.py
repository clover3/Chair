import math
from dataclasses import dataclass
from typing import List, Iterable, Callable, Dict, Tuple, Set, Any
import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils
from transformers import TFBertMainLayer, shape_list, BertConfig, TFBertForSequenceClassification
from transformers.tf_utils import stable_softmax

from list_lib import dict_value_map
from trainer_v2.per_project.transparency.mmp.alignment.align_net_v2 import TFBertLayerFlat
from trainer_v2.per_project.transparency.mmp.probe.probe_common import identify_layers, \
    stop_gradient_for_dict, build_probs_from_tensor_d, combine_qd_mask, get_attn_mask_bias
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


def reshape_per_head_features(out_d, all_head_size):
    bhmd_features = ['query_layer', 'key_layer', 'value_layer']
    bhmd_feature_d = {k: out_d[k] for k in bhmd_features}
    batch_size = shape_list(out_d['query_layer'])[0]

    def reshape_bhmd(tensor):
        t = tf.transpose(tensor, perm=[0, 2, 1, 3])
        t = tf.reshape(tensor=t, shape=(batch_size, -1, all_head_size))
        return t

    bhmd_feature_d = dict_value_map(reshape_bhmd, bhmd_feature_d)
    return bhmd_feature_d


def mean_pool_over_masked(vector, mask):
    """

    :param vector: 3D tensor of [B, M, H]
    :param mask: 2D tensor of [B, M]
    :return:
    """
    mask_f = tf.cast(tf.expand_dims(mask, axis=2), tf.float32)
    t = vector * mask_f
    n_item = tf.reduce_sum(mask_f, axis=1)
    t = tf.reduce_sum(t, axis=1)
    return tf.divide(t, n_item)


def get_emb_concat_feature(bert_main_layer, projection_layer, q_term, d_term):
    q_term_emb = bert_main_layer.embeddings(input_ids=q_term)
    d_term_emb = bert_main_layer.embeddings(input_ids=d_term)
    t = tf.concat([q_term_emb, d_term_emb], axis=2)[:, 0, :]
    k = "emb_concat"
    t_stop = tf.stop_gradient(t, name=f"{k}_stop_gradient")
    emb_concat_probe = projection_layer('emb_concat')(t_stop)
    return emb_concat_probe


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
        y_pred_score = output_d[self.pred_parent][self.pred_key] # [B, 1]
        label = output_d["label"]  # [B, 1]
        is_valid = output_d["is_valid"]  # [B, 1]
        sample_weight_per_label = tf.cast(is_valid, tf.float32) # [B, 1]

        y_pred = y_pred_score > 0  # [B, 1]
        label_b = tf.cast(label, tf.bool) # [B, 1]
        is_correct = tf.cast(tf.equal(label_b, y_pred), tf.float32) # [B, 1]
        n_valid_correct = tf.reduce_sum(is_correct * sample_weight_per_label)  # scalar
        self.correct.assign_add(n_valid_correct)
        n_valid = tf.reduce_sum(sample_weight_per_label)
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
        align_probe = output_d['align_probe'] # Each value has [B, 1]
        label = output_d['label']  # [B, 1]
        is_valid = output_d['is_valid']  # [B, 1]
        sample_weight = tf.cast(is_valid, tf.float32) # [B, 1]

        loss_d = {}
        for k, pred in align_probe.items():
            losses = self.base_loss_fn(label, pred, sample_weight=sample_weight)
            loss_d[f"align/{k}_loss"] = tf.reduce_mean(losses)

        loss_d_values = loss_d.values()
        loss = tf.reduce_sum(list(loss_d_values))
        return loss
