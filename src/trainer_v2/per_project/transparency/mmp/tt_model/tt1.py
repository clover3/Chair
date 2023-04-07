from typing import Dict

import tensorflow as tf
from tensorflow import keras

from models.transformer.bert_common_v2 import get_shape_list2
from trainer_v2.bert_for_tf2 import BertModelLayer
from trainer_v2.custom_loop.modeling_common.bert_common import BERT_CLS
from trainer_v2.custom_loop.modeling_common.network_utils import vector_three_feature


class InputShapeConfigTT:
    max_terms: int
    max_subword_per_word: int


class InputShapeConfigTT100_4(InputShapeConfigTT):
    max_terms = 100
    max_subword_per_word = 4


class ScoringLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ScoringLayer, self).__init__(**kwargs)
        self.k1 = 0.1
        self.b = 1.2
        self.avdl = 50

    def call(self, inputs):
        q_rep, d_rep, q_bow, d_bow = inputs
        batch_size, num_window, _ = get_shape_list2(q_bow['input_ids'])

        def get_null_mask(input_ids):
            all_zero = tf.reduce_all(input_ids == 0, axis=2)
            m = tf.cast(tf.logical_not(all_zero), tf.float32)
            return tf.expand_dims(m, axis=2)

        def check_exact_match(q_input_ids, d_input_ids):
            q_repeat = tf.tile(tf.expand_dims(q_input_ids, axis=2), [1, 1, num_window, 1])  # [B, M, M, W]
            d_repeat = tf.tile(tf.expand_dims(d_input_ids, axis=1), [1, num_window, 1, 1])
            em = tf.reduce_all(tf.equal(q_repeat, d_repeat), axis=3)  # [B, M, M]
            return tf.cast(em, tf.float32)

        # [B, M]
        def get_expanded_doc_tf(d_expanded_tf_rel, q_bow, d_bow):
            em_f = check_exact_match(q_bow['input_ids'], d_bow['input_ids'])  # exact match as float (0.0 or 1.0)
            d_tf = d_expanded_tf_rel + em_f
            tf_multiplier = tf.tile(tf.expand_dims(d_bow['tfs'], axis=1), [1, num_window, 1])
            tf_multiplier = tf.cast(tf_multiplier, tf.float32)
            expanded_term_df = tf.reduce_sum(d_tf * tf_multiplier, axis=2)
            return expanded_term_df

        def bm25_like(d_tf, dl):
            denom = d_tf + d_tf * self.k1
            nom = d_tf + self.k1 * ((1 - self.b) + self.b * dl / self.avdl)
            dtw = denom / (nom + 1e-8)
            qtw = tf.ones([batch_size, num_window])
            return tf.reduce_sum(qtw * dtw, axis=1)

        q_rep = get_null_mask(q_bow['input_ids']) * q_rep
        d_rep = get_null_mask(d_bow['input_ids']) * d_rep
        d_t = tf.transpose(d_rep, [0, 2, 1])
        d_expanded_tf = tf.matmul(q_rep, d_t)  # [B, M, M]
        d_tf = get_expanded_doc_tf(d_expanded_tf, q_bow, d_bow)  # [B, M]
        dl = tf.cast(tf.reduce_sum(d_bow['tfs'], axis=1, keepdims=True), tf.float32)
        s = bm25_like(d_tf, dl)
        return s




class TranslationTable:
    def __init__(self, bert_params, config: InputShapeConfigTT):
        l_bert = BertModelLayer.from_params(bert_params, name="bert")
        pooler = tf.keras.layers.Dense(bert_params.hidden_size, activation=tf.nn.tanh, name="bert/pooler/dense")
        bert_cls = BERT_CLS(l_bert, pooler)
        window_len = config.max_subword_per_word
        num_window = config.max_terms
        role_list = ["q", "d1", "d2"]

        inputs = []
        bow_reps = {}
        for role in role_list:
            input_ids = keras.layers.Input(shape=(num_window * window_len,), dtype='int32', name=f"{role}_input_ids")
            input_ids_stacked = tf.reshape(
                input_ids, [-1, num_window, window_len],
                name=f"{role}_input_ids_stacked")
            tfs = keras.layers.Input(shape=(num_window,), dtype='int32', name=f"{role}_tfs")
            bow_reps[role] = {'input_ids': input_ids_stacked, 'tfs': tfs}
            inputs.append(input_ids)
            inputs.append(tfs)

        batch_size, _, _ = get_shape_list2(bow_reps['q']['input_ids'])
        all_input_ids = tf.concat([bow_reps[role]['input_ids'] for role in role_list], axis=0)  # [B, M * 3, W]
        all_input_ids_flat = tf.reshape(all_input_ids, [-1, window_len])

        # segment input_ids, (possibly append cls)
        dummy_segment_ids = tf.zeros_like(all_input_ids_flat, tf.int32)
        cls = bert_cls.apply([all_input_ids_flat, dummy_segment_ids])
        cls_out = tf.reshape(cls, [batch_size, 3, num_window, bert_params.hidden_size], name="cls_out")
        interaction_rep_size = bert_params.hidden_size
        item_rep = tf.keras.layers.Dense(interaction_rep_size)(cls_out)
        q_rep = item_rep[:, 0, :, :]   # [B, M, W]
        d1_rep = item_rep[:, 1, :, :]
        d2_rep = item_rep[:, 2, :, :]
        get_doc_score = ScoringLayer()
        s1 = get_doc_score([q_rep, d1_rep, bow_reps['q'], bow_reps['d1']])
        s2 = get_doc_score([q_rep, d2_rep, bow_reps['q'], bow_reps['d2']])
        loss = tf.maximum(1 - (s1 - s2), 0)
        output = [(s1, s2), loss]
        model = keras.Model(inputs=inputs, outputs=output, name="bow_translation_table")
        self.loss = loss
        self.model: keras.Model = model
        self.bert_cls = bert_cls
