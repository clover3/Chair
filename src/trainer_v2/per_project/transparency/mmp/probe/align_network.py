from typing import Dict, Any

import tensorflow as tf
from keras.utils import losses_utils
from transformers import shape_list

import trainer_v2.per_project.transparency.mmp.probe.probe_common
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.probe.probe_common import combine_qd_mask, build_paired_inputs_concat, \
    compute_probe_loss, ProbeMAE, ProbePairwiseAcc, build_probs_from_tensor_d, build_probe_from_layer_features, \
    identify_layers, TFBertLayerFlat
from tf_util.lib.tf_funcs import reduce_max, find_layer

Metric = trainer_v2.per_project.transparency.mmp.probe.probe_common.Metric


def get_attn_mask_bias(attn_mask, dtype):
    one_cst = tf.constant(1.0, dtype=dtype)
    ten_thousand_cst = tf.constant(-10000.0, dtype=dtype)
    attn_mask = tf.cast(attn_mask, dtype=dtype)
    attn_mask_bias = tf.multiply(tf.subtract(one_cst, attn_mask), ten_thousand_cst)
    return attn_mask_bias



def build_align_features(out_d, qd_target_mask, q_mask, d_mask) -> Dict[str, tf.Tensor]:
    """

    :param out_d:
    :param qd_target_mask: [B, M, M]
    :param q_mask:
    :param d_mask:
    :return: Dictionary of tensors as values
    each tensor has shape of [B, ?]
    """
    # Features 1) Attention score between q_term/d_term
    # Features 2) Attention prob between q_term/d_term
    # Features 3) Global attention output
    # Features 4)
    mask = tf.cast(tf.expand_dims(qd_target_mask, axis=1), tf.float32)
    q_mask_ex = tf.cast(tf.expand_dims(q_mask, axis=2), tf.float32)
    d_mask_ex = tf.cast(tf.expand_dims(d_mask, axis=2), tf.float32)

    # [B, H]
    # For each batch,head, select scores that are from document term to query term
    attn_scores_based = reduce_max(out_d['attention_scores'] * mask, [2, 2])
    attn_probs_based = reduce_max(out_d['attention_probs'] * mask, [2, 2])

    out_feature_set = {}
    # add last dimension which makes it feature of width 12, which is number of heads
    out_feature_set['attn_scores'] = attn_scores_based
    out_feature_set['attn_probs'] = attn_probs_based

    # For features that has shape of [B, M, D], features are built by comparing ablated feature*
    # ablated features are the ones that only have attention vector from the selected document terms
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
        # Select the feature at the query term location
        base_out = tf.reduce_sum(out_d[key] * q_mask_ex, axis=1)
        alt_out = tf.reduce_sum(out_d[key_tm] * q_mask_ex, axis=1)

        all_linear_features_vectors.append(base_out)
        all_linear_features_vectors.append(alt_out)

        pair_concat = tf.concat([base_out, alt_out], axis=1)
        out_feature_set[key + "_paired"] = pair_concat

    out_feature_set['all_concat'] = tf.concat(all_linear_features_vectors, axis=1)

    return out_feature_set


class AlignAcc(trainer_v2.per_project.transparency.mmp.probe.probe_common.Metric):
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
        is_valid_f = tf.cast(is_valid, tf.float32)  # [B, 1]

        y_pred = y_pred_score > 0  # [B, 1]
        label_b = tf.cast(label, tf.bool) # [B, 1]
        is_correct = tf.cast(tf.equal(label_b, y_pred), tf.float32) # [B, 1]

        n_valid_correct = tf.reduce_sum(is_correct * is_valid_f)  # scalar
        self.correct.assign_add(n_valid_correct)
        n_valid = tf.reduce_sum(is_valid_f)
        self.count.assign_add(n_valid)

    def result(self):
        return self.correct / self.count

    def reset_state(self):
        self.correct.assign(0.0)
        self.count.assign(0.0)


class AddLosses(tf.keras.losses.Loss):
    def __init__(self, inner_losses: tf.keras.losses.Loss, name="AddLosses"):
        super().__init__(name=name)
        self.inner_losses = inner_losses

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, output_d):
        return tf.reduce_sum([loss(output_d) for loss in self.inner_losses])


class ProbeLossOnSeq(tf.keras.losses.Loss):
    def __init__(self, name='logit_probe_on_align_feature'):
        super().__init__(name=name)
        delta = 0.5
        self.base_loss_fn = tf.keras.losses.Huber(delta=delta, reduction=losses_utils.ReductionV2.NONE)


    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, output_d):
        logit_probe_on_align_feature = output_d['logit_probe_on_align_feature']
        logits = output_d['logits']
        return compute_probe_loss(logits, logit_probe_on_align_feature, self.base_loss_fn)


class ProbeLossFromDict(tf.keras.losses.Loss):
    def __init__(self, name='probe_loss_from_dict'):
        super().__init__(name=name)
        delta = 0.5
        self.base_loss_fn = tf.keras.losses.Huber(delta=delta, reduction=losses_utils.ReductionV2.NONE)

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, output_d):
        input_mask = output_d['input_mask']
        probe_on_hidden = output_d['logit_probe_on_hidden']
        probe_on_attn_like = output_d['logit_probe_on_attn_like']
        logits = output_d['logits']
        sample_weight = tf.cast(input_mask, tf.float32)
        def loss_fn(target, pred):
            return self.base_loss_fn(target, pred, sample_weight=sample_weight)

        all_d = {}
        all_d.update(probe_on_hidden)
        all_d.update(probe_on_attn_like)
        return compute_probe_loss(logits, all_d, loss_fn)


class AlignLossFromDict(tf.keras.losses.Loss):
    def __init__(self, name='align_loss_from_dict'):
        super().__init__(name=name)
        # delta = 0.5
        # self.base_loss_fn = tf.keras.losses.Huber(delta=delta, reduction=losses_utils.ReductionV2.NONE)
        self.base_loss_fn = tf.keras.losses.BinaryCrossentropy(
            from_logits=True,
            reduction=losses_utils.ReductionV2.NONE)

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, output_d):
        align_probe = output_d['align_probe']
        label = output_d['label']  # [B, 1]

        is_valid = output_d['is_valid']
        sample_weight = tf.cast(is_valid, tf.float32)

        def loss_fn(pred):
            # pred has [B, 1]
            losses = self.base_loss_fn(label, pred, sample_weight=sample_weight)
            return tf.reduce_mean(losses)

        loss_d = {}
        for k, v in align_probe.items():
            loss_d[f"align/{k}_loss"] = loss_fn(v)

        loss_d_values = loss_d.values()
        loss = tf.reduce_sum(list(loss_d_values))
        return loss


class BertGAlignLayer(tf.keras.layers.Layer):
    def __init__(self):
        self.bert_flat = TFBertLayerFlat(bert_config, layers_d)
        self.n_out_dim = 1

    def call(self, input_d):
        input_ids = input_d["input_ids"]
        segment_ids = input_d["token_type_ids"]
        q_term_mask = input_d["q_term_mask"]
        d_term_mask = input_d["d_term_mask"]
        label = input_d["label"]
        is_valid = input_d["is_valid"]
        # Part 2. Encode with bert_main_layer/classifier

        qd_target_mask = combine_qd_mask(q_term_mask, d_term_mask)
        c_log.info("bert_main_layer")
        bert_outputs = self.bert_main_layer(
            [input_ids, segment_ids],
            output_attentions=True,
            output_hidden_states=True)
        logits = self.orig_classifier(bert_outputs.pooler_output)
        logits = tf.stop_gradient(logits, "logits_stop_gradient")
        attn = bert_outputs.attentions
        hidden = bert_outputs.hidden_states
        hidden = [tf.stop_gradient(h) for h in hidden]

        # Part 3. Compute Layer-inner features
        input_mask = tf.cast(tf.not_equal(input_ids, 0), tf.int32)
        input_mask_shape = shape_list(input_mask)
        attn_mask = tf.reshape(
            input_mask, (input_mask_shape[0], 1, 1, input_mask_shape[1])
        )
        embedding = hidden[0]
        dtype = embedding.dtype
        attn_mask_bias = get_attn_mask_bias(attn_mask, dtype)

        per_layer_feature_tensors = self.bert_flat(embedding, attn_mask_bias, qd_target_mask)

        # Part 4. Probes
        hidden_tensor_d = {}
        for i, hidden_layer in enumerate(hidden):
            if 3 < i < 10:
                continue
            key = 'layer_{}'.format(i)
            hidden_tensor_d[key] = hidden_layer

        probe_on_hidden: Dict[str, tf.Tensor] = build_probs_from_tensor_d(hidden_tensor_d)

        probe_on_attn_like = build_probe_from_layer_features(
            per_layer_feature_tensors, self.bert_config.hidden_size, self.n_out_dim)
        #  dd
        align_feature_d = build_align_features(per_layer_feature_tensors,
                             qd_target_mask, q_term_mask, d_term_mask)

        align_feature_d_l = {"L_" + k: v for k, v in align_feature_d.items()}
        logit_probe_on_align_feature = build_probs_from_tensor_d(align_feature_d_l)
        align_probe: Dict[str, tf.Tensor] = build_probs_from_tensor_d(align_feature_d)
        output_d = {
            "logit_probe_on_hidden": probe_on_hidden,  # Dict[str, tf.Tensor]
            "logit_probe_on_attn_like": probe_on_attn_like,
            "logit_probe_on_align_feature": logit_probe_on_align_feature,
            "align_probe": align_probe,
            "logits": logits,
            "input_mask": input_mask,
            "q_term_mask": q_term_mask,
            "d_term_mask": d_term_mask,
            "label": label,
            "is_valid": is_valid,
        }
        return output_d


class GAlignNetwork:
    def __init__(self, bert_model):
        n_out_dim = 1
        target_layer_no = 0

        # We skip dropout
        bert_main_layer = find_layer(bert_model, "bert")
        orig_classifier = find_layer(bert_model, "classifier")
        bert_config = bert_main_layer._config
        layers_d = identify_layers(bert_main_layer, target_layer_no)

        # Part 1. Build inputs
        keys = ["input_ids", "token_type_ids", "q_term_mask", "d_term_mask", "label", "is_valid"]
        input_concat_d, inputs = build_paired_inputs_concat(keys)

        input_ids = input_concat_d["input_ids"]
        segment_ids = input_concat_d["token_type_ids"]
        q_term_mask = input_concat_d["q_term_mask"]
        d_term_mask = input_concat_d["d_term_mask"]
        label = input_concat_d["label"]
        is_valid = input_concat_d["is_valid"]

        # Part 2. Encode with bert_main_layer/classifier

        qd_target_mask = combine_qd_mask(q_term_mask, d_term_mask)
        c_log.info("bert_main_layer")
        bert_input = {
            'input_ids': input_ids,
            'token_type_ids': segment_ids
        }
        bert_outputs = bert_main_layer(
            bert_input,
            output_attentions=True,
            output_hidden_states=True)
        logits = orig_classifier(bert_outputs.pooler_output)
        logits = tf.stop_gradient(logits, "logits_stop_gradient")
        attn = bert_outputs.attentions
        hidden = bert_outputs.hidden_states
        hidden = [tf.stop_gradient(h) for h in hidden]

        # Part 3. Compute Layer-inner features
        input_mask = tf.cast(tf.not_equal(input_ids, 0), tf.int32)
        input_mask_shape = shape_list(input_mask)
        attn_mask = tf.reshape(
            input_mask, (input_mask_shape[0], 1, 1, input_mask_shape[1])
        )
        embedding = hidden[0]
        dtype = embedding.dtype
        attn_mask_bias = get_attn_mask_bias(attn_mask, dtype)

        bert_flat = TFBertLayerFlat(bert_config, layers_d)
        per_layer_feature_tensors = bert_flat(embedding, attn_mask_bias, qd_target_mask)

        # Part 4. Probes
        hidden_tensor_d = {}
        for i, hidden_layer in enumerate(hidden):
            if 3 < i < 10:
                continue
            key = 'layer_{}'.format(i)
            hidden_tensor_d[key] = hidden_layer

        probe_on_hidden: Dict[str, tf.Tensor] = build_probs_from_tensor_d(hidden_tensor_d)

        probe_on_attn_like = build_probe_from_layer_features(
            per_layer_feature_tensors, bert_config.hidden_size, n_out_dim)
        #  dd
        align_feature_d = build_align_features(per_layer_feature_tensors,
                             qd_target_mask, q_term_mask, d_term_mask)

        align_feature_d_l = {"L_" + k: v for k, v in align_feature_d.items()}
        logit_probe_on_align_feature = build_probs_from_tensor_d(align_feature_d_l)
        align_probe: Dict[str, tf.Tensor] = build_probs_from_tensor_d(align_feature_d)
        output_d = {
            "logit_probe_on_hidden": probe_on_hidden,  # Dict[str, tf.Tensor]
            "logit_probe_on_attn_like": probe_on_attn_like,
            "logit_probe_on_align_feature": logit_probe_on_align_feature,
            "align_probe": align_probe,
            "logits": logits,
            "input_mask": input_mask,
            "q_term_mask": q_term_mask,
            "d_term_mask": d_term_mask,
            "label": label,
            "is_valid": is_valid,
        }

        self.probe_model_output: Dict[str, Any] = output_d
        self.model = tf.keras.models.Model(
            inputs=inputs, outputs=self.probe_model_output)


    def get_probe_metrics(self) -> Dict[str, Metric]:
        output_d = {}
        target_key = 'logits'
        short_mapping = {
            "logit_probe_on_hidden": "LH",
            "logit_probe_on_attn_like": "LA",
            "logit_probe_on_align_feature": "LAF"
        }
        for probe_group in ["logit_probe_on_hidden",
                            "logit_probe_on_attn_like",
                            "logit_probe_on_align_feature"]:
            d = self.probe_model_output[probe_group]
            # with tf.name_scope(probe_group):
            for pred_key, out_tensor in d.items():
                group_short = short_mapping[probe_group]
                metric_name = f"{group_short}/{pred_key}/MAE"
                metric = ProbeMAE(target_key, probe_group, pred_key, name=metric_name)
                output_d[metric_name] = metric

                metric_name = f"{group_short}/{pred_key}/PairAcc"
                metric = ProbePairwiseAcc(target_key, probe_group, pred_key, name=metric_name)
                output_d[metric_name] = metric

        return output_d

    def get_align_metrics(self) -> Dict[str, Metric]:
        output_d = {}
        d = self.probe_model_output["align_probe"]
        for pred_key, out_tensor in d.items():
            metric_name = f"align/{pred_key}"
            metric = AlignAcc("align_probe", pred_key, name=metric_name)
            output_d[metric_name] = metric
        return output_d