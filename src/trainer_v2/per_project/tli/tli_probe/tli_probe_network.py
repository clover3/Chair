import tensorflow as tf
from keras.utils import losses_utils
from transformers import TFBertMainLayer, PretrainedConfig, BertConfig
from transformers.tf_utils import shape_list
from typing import Dict, Any

from cpath import get_bert_config_path
from models.transformer.bert_common_v2 import create_attention_mask_from_input_mask, get_shape_list
from trainer_v2.bert_for_tf2 import BertModelLayer
from trainer_v2.chair_logging import c_log
from tf_util.lib.tf_funcs import find_layer
from trainer_v2.per_project.transparency.mmp.probe.probe_common import identify_layers, TFBertLayerFlat, \
    build_probs_from_tensor_d, build_probe_from_layer_features, compute_probe_loss


def convert_label(tli_label_flat, input_ids, segment_ids, sep_id=102):
    n_label = 3
    input_ids_shape = shape_list(input_ids)
    B = input_ids_shape[0]
    tli_label_3d = tf.reshape(tli_label_flat, [B, n_label, -1])
    tli_label_3d = tf.transpose(tli_label_3d, [0, 2, 1])

    t_pos = 0.7
    t_neg = 0.3

    label_pos = tf.less(t_pos, tli_label_3d)
    label_neg = tf.less(tli_label_3d, t_neg)
    no_label = tf.logical_or(label_pos, label_neg)

    some_label = tf.logical_not(no_label)

    not_sep = tf.not_equal(sep_id, input_ids)
    is_second = tf.equal(segment_ids, 1)
    is_valid = tf.logical_and(not_sep, is_second)
    is_valid = tf.logical_and(some_label, tf.expand_dims(is_valid, axis=2))

    tli_label = tf.cast(label_pos, tf.int32)
    is_valid_i = tf.cast(is_valid, tf.int32)
    return tli_label, is_valid_i


class ProbeBCE(tf.keras.losses.Loss):
    def __init__(self, name='probe_loss_from_dict'):
        super().__init__(name=name)
        self.base_loss_fn = tf.keras.losses.BinaryCrossentropy(
            from_logits=True,
            reduction=losses_utils.ReductionV2.NONE)

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, output_d):
        def ex(t):
            return tf.expand_dims(t, axis=-1)
        tli_categorical_label = ex(output_d['tli_categorical_label'])
        tli_label_mask = ex(output_d['tli_label_mask'])
        probe_on_hidden = output_d['probe_on_hidden']

        sample_weight = tf.cast(tli_label_mask, tf.float32) # [B, 3, M]
        all_d = {}
        all_d.update(probe_on_hidden)

        loss_d = {}
        for k, v in all_d.items():
            v_ex = ex(v)
            losses = self.base_loss_fn(tli_categorical_label, v_ex, sample_weight=sample_weight)

            losses = tf.reduce_sum(tf.reduce_sum(losses, axis=2), axis=1)
            loss = tf.reduce_mean(losses)
            loss_d[f"probe/{k}_loss"] = loss

        loss_d_values = loss_d.values()
        loss = tf.reduce_sum(list(loss_d_values))
        return loss


class ProbeAcc(tf.keras.metrics.Metric):
    def __init__(self, target_key, pred_parent, pred_key, mask_key, name, **kwargs):
        super(ProbeAcc, self).__init__(name=name, **kwargs)
        self.mae = self.add_weight(name='mae', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        self.target_key = target_key
        self.pred_parent = pred_parent
        self.pred_key = pred_key
        self.mask_key = mask_key

    def update_state(self, output_d, _sample_weight=None):
        y_true = output_d[self.target_key]
        y_pred = output_d[self.pred_parent][self.pred_key]
        y_pred_prob = tf.nn.sigmoid(y_pred)
        y_pred_label = tf.cast(tf.less(0.5, y_pred_prob), tf.int32)
        sample_mask = output_d[self.mask_key]
        sample_weight = tf.cast(sample_mask, tf.float32)

        correct_f = tf.cast(tf.equal(y_true, y_pred_label), tf.float32)
        correct_f = correct_f * sample_weight
        correct_sum = tf.reduce_sum(correct_f)
        self.mae.assign_add(correct_sum)
        n_valid = tf.reduce_sum(tf.cast(sample_weight, tf.float32))
        self.count.assign_add(n_valid)

    def result(self):
        return self.mae / self.count

    def reset_state(self):
        self.mae.assign(0.0)
        self.count.assign(0.0)



class TliProbe:
    def __init__(self, model_config, bert_params, convert_label_fn=convert_label):
        num_classes = model_config.num_classes
        max_seq_len = model_config.max_seq_length

        n_out_dim = num_classes
        l_bert = BertModelLayer.from_params(bert_params, name="bert")
        pooler = tf.keras.layers.Dense(bert_params.hidden_size, activation=tf.nn.tanh, name="bert/pooler/dense")
        classifier = tf.keras.layers.Dense(n_out_dim, activation=tf.nn.softmax)

        input_ids = tf.keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32, name="input_ids")
        segment_ids = tf.keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32, name="segment_ids")
        label_ids = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name="label_ids")
        tli_label = tf.keras.layers.Input(shape=(max_seq_len * n_out_dim,), dtype=tf.float32, name="tli_label")
        tli_categorical_label, tli_label_mask = convert_label_fn(tli_label, input_ids, segment_ids)

        inputs = [input_ids, segment_ids, label_ids, tli_label]


        c_log.info("bert_main_layer")
        bert_output = l_bert([input_ids, segment_ids])
        seq_out = bert_output[-1]
        first_token = seq_out[:, 0, :]
        pooled = pooler(first_token)
        cls_probs = classifier(pooled)
        hidden = [tf.stop_gradient(h) for h in bert_output]

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

        c_log.info("probe_on_hidden")
        hidden_tensor_d = {}
        for i, hidden_layer in enumerate(hidden):
            if 3 < i < 10:
                continue
            key = 'layer_{}'.format(i)
            hidden_tensor_d[key] = hidden_layer

        probe_on_hidden: Dict[str, tf.Tensor] = build_probs_from_tensor_d(hidden_tensor_d, n_out_dim)

        self.probe_model_output: Dict[str, Any] = {
            "probe_on_hidden": probe_on_hidden,  # Dict[str, tf.Tensor]
            "cls_probs": cls_probs,
            'label_ids': label_ids,
            "tli_categorical_label": tli_categorical_label,
            "tli_label_mask": tli_label_mask,
            "input_mask": input_mask,
        }
        self.model = tf.keras.models.Model(inputs=inputs, outputs=self.probe_model_output)

    def get_attn_mask_bias(self, attn_mask, dtype):
        one_cst = tf.constant(1.0, dtype=dtype)
        ten_thousand_cst = tf.constant(-10000.0, dtype=dtype)
        attn_mask = tf.cast(attn_mask, dtype=dtype)
        attn_mask_bias = tf.multiply(tf.subtract(one_cst, attn_mask), ten_thousand_cst)
        return attn_mask_bias

    def get_probe_metrics(self) -> Dict[str, tf.keras.metrics.Metric]:
        output_d = {}
        target_key = 'tli_categorical_label'
        short_mapping = {
            "probe_on_hidden": "hidden",
        }
        for probe_group in ["probe_on_hidden", ]:
            d = self.probe_model_output[probe_group]
            for pred_key, out_tensor in d.items():
                group_short = short_mapping[probe_group]
                metric_name = f"{group_short}/{pred_key} Acc"
                metric = ProbeAcc(
                    target_key, probe_group, pred_key, "tli_label_mask", name=metric_name)
                output_d[metric_name] = metric
        return output_d


