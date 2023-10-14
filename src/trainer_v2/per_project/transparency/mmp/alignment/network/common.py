import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils
from transformers import shape_list

from list_lib import dict_value_map
from models.transformer.bert_common_v2 import get_shape_list
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.tt_model.net_common import pairwise_hinge


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


def build_align_acc_dict(d):
    output_d = {}
    for pred_key, out_tensor in d.items():
        metric_name = f"align/{pred_key}"
        metric = AlignAcc("align_probe", pred_key, name=metric_name)
        output_d[metric_name] = metric
    return output_d


def build_align_acc_dict_pairwise(d):
    output_d = {}
    for pred_key, out_tensor in d.items():
        metric_name = f"align/{pred_key}"
        metric = AlignAccPairwise("align_probe", pred_key, name=metric_name)
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


class AlignAccPairwise(tf.keras.metrics.Metric):
    def __init__(self, pred_parent, pred_key, name, **kwargs):
        super(AlignAccPairwise, self).__init__(name=name, **kwargs)
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
        self.pred_parent = pred_parent
        self.pred_key = pred_key

    def update_state(self, output_d, _sample_weight=None):
        y_pred_score = output_d[self.pred_parent][self.pred_key]  # [B*2, 1]
        pred_pos, pred_neg = split_concat(y_pred_score)
        is_correct = tf.cast(tf.less(pred_neg, pred_pos), tf.float32)  # [B, 1]
        n_correct = tf.reduce_sum(is_correct)  # scalar
        self.correct.assign_add(n_correct)
        n_valid = tf.reduce_sum(tf.ones_like(is_correct))
        self.count.assign_add(n_valid)

    def result(self):
        return self.correct / self.count

    def reset_state(self):
        self.correct.assign(0.0)
        self.count.assign(0.0)


class AlignLossFromDict(tf.keras.losses.Loss):
    def __init__(self, name='align_loss_from_dict'):
        super().__init__(name=name)
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


def split_concat(tensor):
    B, D = get_shape_list(tensor)
    t_stacked = tf.reshape(tensor, [-1, 2, D])
    return t_stacked[:, 0, :], t_stacked[:, 1, :]


class AlignLossFromDictPairConcat(tf.keras.losses.Loss):
    def __init__(self, name='align_loss_from_dict'):
        super().__init__(name=name)

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, output_d):
        align_probe = output_d['align_probe']  # Each value has [B, 1]
        loss_d = {}
        for k, pred in align_probe.items():
            pos_pred, neg_pred = split_concat(pred)
            losses = pairwise_hinge(pos_pred, neg_pred)
            loss_d[f"align/{k}_loss"] = tf.reduce_mean(losses)

        loss_d_values = loss_d.values()
        loss = tf.reduce_sum(list(loss_d_values))
        return loss



def define_pairwise_galign_inputs(max_term_len, tokenizer):
    cls_id = tokenizer.vocab["[CLS]"]
    sep_id = tokenizer.vocab["[SEP]"]

    q_term_flat, d_term_flat, inputs = define_pairwise_input_and_stack_flat(max_term_len)
    d_term_mask, input_ids, q_term_mask, token_type_ids = form_input_ids_segment_ids(
        q_term_flat, d_term_flat,
        cls_id, sep_id,
        max_term_len)
    return input_ids, token_type_ids, inputs


def define_pointwise_galign_inputs(max_term_len, tokenizer):
    q_term = tf.keras.layers.Input(shape=(max_term_len,), dtype='int32', name="q_term")
    d_term = tf.keras.layers.Input(shape=(max_term_len,), dtype='int32', name="d_term")
    inputs = [q_term, d_term]
    d_term_mask, input_ids, q_term_mask, token_type_ids = build_input_ids_segment_ids(
        q_term, d_term, max_term_len, tokenizer)

    return input_ids, token_type_ids, inputs


def define_pairwise_input_and_stack_flat(max_term_len):
    q_term_raw = tf.keras.layers.Input(shape=(max_term_len,), dtype='int32', name="q_term")
    d_term_pos = tf.keras.layers.Input(shape=(max_term_len,), dtype='int32', name="d_term_pos")
    d_term_neg = tf.keras.layers.Input(shape=(max_term_len,), dtype='int32', name="d_term_neg")
    inputs = [q_term_raw, d_term_pos, d_term_neg]
    d_term_stack = tf.stack([d_term_pos, d_term_neg], axis=1)  # [Batch, 2, Max term len]
    d_term_flat = tf.reshape(d_term_stack, [-1, max_term_len])
    q_term_ex = tf.expand_dims(q_term_raw, axis=1)
    q_term_stack = tf.tile(q_term_ex, [1, 2, 1])
    q_term_flat = tf.reshape(q_term_stack, [-1, max_term_len])
    return q_term_flat, d_term_flat, inputs


def form_input_ids_segment_ids(q_term_flat, d_term_flat, cls_id, sep_id, max_term_len):
    B, _ = shape_list(q_term_flat)
    CLS = tf.ones([B, 1], tf.int32) * cls_id
    SEP = tf.ones([B, 1], tf.int32) * sep_id
    ZERO = tf.zeros([B, 1], tf.int32)
    input_ids = tf.concat([CLS, q_term_flat, SEP, d_term_flat, SEP], axis=1)
    q_term_mask = tf.concat([ZERO, tf.ones_like(q_term_flat, tf.int32), ZERO,
                             tf.zeros_like(d_term_flat, tf.int32), ZERO], axis=1)
    d_term_mask = tf.concat([ZERO, tf.zeros_like(q_term_flat, tf.int32), ZERO,
                             tf.ones_like(d_term_flat, tf.int32), ZERO], axis=1)
    seg1_len = max_term_len + 2
    seg2_len = max_term_len + 1
    token_type_ids_row = [0] * seg1_len + [1] * seg2_len
    token_type_ids = tf.tile(tf.expand_dims(token_type_ids_row, 0), [B, 1])
    return d_term_mask, input_ids, q_term_mask, token_type_ids


def build_input_ids_segment_ids(q_term, d_term, max_term_len, tokenizer):
    cls_id = tokenizer.vocab["[CLS]"]
    sep_id = tokenizer.vocab["[SEP]"]
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
    return d_term_mask, input_ids, q_term_mask, token_type_ids