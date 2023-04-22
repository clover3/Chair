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



def tf_print(*args):
    # tf.print(*args)
    pass


class TermEncoder(tf.keras.layers.Layer):
    def __init__(self, bert_params):
        super(TermEncoder, self).__init__()
        self.interaction_rep_size = bert_params.hidden_size

        l_bert = BertModelLayer.from_params(bert_params, name="bert")
        pooler = tf.keras.layers.Dense(bert_params.hidden_size, activation=tf.nn.tanh, name="bert/pooler/dense")
        self.bert_cls = BERT_CLS(l_bert, pooler)
        self.l_bert = l_bert
        self.dense1 = tf.keras.layers.Dense(self.interaction_rep_size,
                                            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                                stddev=1e-3)
                                            )
        # self.layernorm = tf.keras.layers.LayerNormalization(name="layernorm")

    def call(self, inputs):
        input_ids = inputs  # [B, max_term, term_len]
        batch_size, num_terms, window_len = get_shape_list2(input_ids)
        input_ids_flat = tf.reshape(input_ids, [-1, window_len])
        dummy_segment_ids = tf.zeros_like(input_ids_flat, tf.int32)
        cls = self.bert_cls.apply([input_ids_flat, dummy_segment_ids])
        item_rep = self.dense1(cls)
        item_rep = tf.reshape(item_rep, [batch_size, num_terms, self.interaction_rep_size])
        return item_rep


class DummyTermEncoder(tf.keras.layers.Layer):
    def __init__(self, bert_params):
        super(DummyTermEncoder, self).__init__()
        self.interaction_rep_size = bert_params.hidden_size
        l_bert = BertModelLayer.from_params(bert_params, name="bert")
        pooler = tf.keras.layers.Dense(bert_params.hidden_size, activation=tf.nn.tanh, name="bert/pooler/dense")
        self.bert_cls = BERT_CLS(l_bert, pooler)


    def call(self, inputs):
        input_ids = inputs  # [B, max_term, term_len]
        batch_size, num_terms, window_len = get_shape_list2(input_ids)
        item_rep = tf.zeros([batch_size, num_terms, self.interaction_rep_size])
        return item_rep



class ScoringLayer(tf.keras.layers.Layer):
    def __init__(self, interaction_rep_size, **kwargs):
        super(ScoringLayer, self).__init__(**kwargs)
        self.k1 = 0.1
        self.b = 1.2
        self.avdl = 50
        self.alpha = 1 / interaction_rep_size
        self.beta = 0.1
        print(f": Alpha={self.alpha} beta={self.beta}")

    def call(self, inputs):
        q_rep, d_rep, q_bow, d_bow = inputs
        batch_size, num_window, _ = get_shape_list2(q_bow['input_ids'])
        batch_size, num_window, hidden_size = get_shape_list2(q_rep)
        if "qtw" in q_bow:
            qtw = q_bow['qtw']
        else:
            qtw = tf.ones([batch_size, num_window])

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
            # tf_print("em_f[0]", em_f[0, :4, :8])
            # tf_print("em_f[1]", em_f[0, :4, :8])
            # tf_print("d_expanded_tf_rel[0] ", d_expanded_tf_rel[0, :4, :8])
            # tf_print("d_expanded_tf_rel[1] ", d_expanded_tf_rel[1, :4, :8])
            d_ex_alpha = d_expanded_tf_rel * self.alpha
            # tf_print("d_ex_alpha[0] ", d_ex_alpha[0, :4, :8])

            d_tf = (d_expanded_tf_rel * self.alpha) * self.beta + em_f * (1-self.beta)
            tf_multiplier = tf.tile(tf.expand_dims(d_bow['tfs'], axis=1), [1, num_window, 1])
            tf_multiplier = tf.cast(tf_multiplier, tf.float32)
            # tf_print("tf_multiplier[0]", tf_multiplier[0, :4, :8])
            d_ex_alpha_reduce = tf.reduce_sum(d_ex_alpha * tf_multiplier, axis=2)
            # tf_print("d_ex_alpha_reduce[0]", d_ex_alpha_reduce[0, :4])
            em_mult_df = tf.reduce_sum(em_f * tf_multiplier, axis=2)
            # tf_print("em_mult_df[0]", em_mult_df[0, :4])
            expanded_term_df = tf.reduce_sum(d_tf * tf_multiplier, axis=2)
            # tf_print("expanded_term_df[0]", expanded_term_df[0, :4])
            return expanded_term_df

        def bm25_like(qtw, d_tf, dl):
            denom = d_tf + d_tf * self.k1
            nom = d_tf + self.k1 * ((1 - self.b) + self.b * dl / self.avdl)
            dtw = denom / (nom + 1e-8)
            return tf.reduce_sum(qtw * dtw, axis=1)

        q_rep = get_null_mask(q_bow['input_ids']) * q_rep
        d_rep = get_null_mask(d_bow['input_ids']) * d_rep
        d_t = tf.transpose(d_rep, [0, 2, 1])
        d_expanded_tf = tf.matmul(q_rep, d_t)  # [B, M, M]
        d_tf = get_expanded_doc_tf(d_expanded_tf, q_bow, d_bow)  # [B, M]
        dl = tf.cast(tf.reduce_sum(d_bow['tfs'], axis=1, keepdims=True), tf.float32)
        s = bm25_like(qtw, d_tf, dl)
        # tf_print("Score", s)
        return s, d_expanded_tf


class ScoringLayer2(tf.keras.layers.Layer):
    def __init__(self, interaction_rep_size, **kwargs):
        super(ScoringLayer2, self).__init__(**kwargs)
        self.k1 = 0.1
        self.b = 1.2
        self.avdl = 50
        self.alpha = 0.2
        self.beta = 0.1
        print(f"beta={self.beta}")

    def call(self, inputs):
        q_rep, d_rep, q_bow, d_bow = inputs
        batch_size, num_window, _ = get_shape_list2(q_bow['input_ids'])
        batch_size, num_window, hidden_size = get_shape_list2(q_rep)
        if "qtw" in q_bow:
            qtw = q_bow['qtw']
        else:
            qtw = tf.ones([batch_size, num_window])

        def get_null_mask(input_ids):  # [B, MaxTerm]
            all_zero = tf.reduce_all(input_ids == 0, axis=2)
            m = tf.cast(tf.logical_not(all_zero), tf.float32)
            return m

        def check_exact_match(q_input_ids, d_input_ids):
            q_repeat = tf.tile(tf.expand_dims(q_input_ids, axis=2), [1, 1, num_window, 1])  # [B, M, M, W]
            d_repeat = tf.tile(tf.expand_dims(d_input_ids, axis=1), [1, num_window, 1, 1])
            em = tf.reduce_all(tf.equal(q_repeat, d_repeat), axis=3)  # [B, M, M]
            return tf.cast(em, tf.float32)

        # [B, M]
        def get_expanded_doc_tf(d_expanded_tf, q_bow, d_bow):
            em_f = check_exact_match(q_bow['input_ids'], d_bow['input_ids'])  # exact match as float (0.0 or 1.0)
            tf_print("d_expanded_tf", d_expanded_tf)
            d_tf = (d_expanded_tf * self.alpha) * self.beta + em_f * (1-self.beta)
            tf_multiplier = tf.tile(tf.expand_dims(d_bow['tfs'], axis=1), [1, num_window, 1])
            tf_multiplier = tf.cast(tf_multiplier, tf.float32)
            expanded_term_df = tf.reduce_sum(d_tf * tf_multiplier, axis=2)
            return expanded_term_df

        def bm25_like(qtw, d_tf, dl):
            denom = d_tf + d_tf * self.k1
            nom = d_tf + self.k1 * ((1 - self.b) + self.b * dl / self.avdl)
            dtw = denom / (nom + 1e-8)
            return tf.reduce_sum(qtw * dtw, axis=1)

        m_q = get_null_mask(q_bow['input_ids'])
        m_d = get_null_mask(d_bow['input_ids'])
        mask = tf.expand_dims(m_q, axis=2) * tf.expand_dims(m_d, axis=1)

        d_t = tf.transpose(d_rep, [0, 2, 1])
        qd_term_dot_output = tf.matmul(q_rep, d_t)  # [B, M, M]
        d_expanded_tf = tf.nn.sigmoid(qd_term_dot_output) * self.alpha
        d_expanded_tf = mask * d_expanded_tf
        d_tf = get_expanded_doc_tf(d_expanded_tf, q_bow, d_bow)  # [B, M]
        dl = tf.cast(tf.reduce_sum(d_bow['tfs'], axis=1, keepdims=True), tf.float32)
        s = bm25_like(qtw, d_tf, dl)
        return s, d_expanded_tf



def contrastive_loss(pos_score, neg_score):
    t = tf.stack([pos_score, neg_score], axis=1)
    t = tf.nn.softmax(t, axis=1)
    t = t[:, 0]
    t = -tf.math.log(t)
    t = tf.clip_by_value(
        t, -1e6, 1e6, name=None
    )
    return t


class TranslationTable:
    def __init__(self, bert_params, config: InputShapeConfigTT):
        term_encoder = TermEncoder(bert_params)
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
        q_rep = term_encoder(bow_reps['q']['input_ids'])
        d1_rep = term_encoder(bow_reps['d1']['input_ids'])
        d2_rep = term_encoder(bow_reps['d2']['input_ids'])
        get_doc_score = ScoringLayer(bert_params.hidden_size)
        s1 = get_doc_score([q_rep, d1_rep, bow_reps['q'], bow_reps['d1']])
        s2 = get_doc_score([q_rep, d2_rep, bow_reps['q'], bow_reps['d2']])
        loss = contrastive_loss(s1, s2)
        output = [(s1, s2), loss]
        model = keras.Model(inputs=inputs, outputs=output, name="bow_translation_table")
        self.loss = loss
        self.model: keras.Model = model
        self.bert_cls = term_encoder.bert_cls


def define_inputs(num_window, window_len):
    role_list = ["q", "d1", "d2"]
    inputs = []
    bow_reps = {}
    for role in role_list:
        input_ids = keras.layers.Input(shape=(num_window * window_len,), dtype='int32', name=f"{role}_input_ids")
        input_ids_stacked = tf.reshape(
            input_ids, [-1, num_window, window_len],
            name=f"{role}_input_ids_stacked")
        tfs = keras.layers.Input(shape=(num_window,), dtype='int32', name=f"{role}_tfs")
        qtw = keras.layers.Input(shape=(num_window,), dtype='float32', name=f"{role}_qtw")
        bow_reps[role] = {
            'input_ids': input_ids_stacked,
            'tfs': tfs,
            'qtw': qtw
        }
        inputs.append(input_ids)
        inputs.append(tfs)
        inputs.append(qtw)
    return bow_reps, inputs


def get_tf_loss(ex_tf):
    gamma = 1 / 5
    ex_tf_abs = tf.abs(ex_tf)
    loss_sum = tf.reduce_sum(tf.reduce_sum(ex_tf_abs, axis=2), axis=1)
    return loss_sum * gamma


class TranslationTableQTW:
    def __init__(self, bert_params, config: InputShapeConfigTT):
        term_encoder = TermEncoder(bert_params)
        window_len = config.max_subword_per_word
        num_window = config.max_terms
        bow_reps, inputs = define_inputs(num_window, window_len)

        batch_size, _, _ = get_shape_list2(bow_reps['q']['input_ids'])
        q_rep = term_encoder(bow_reps['q']['input_ids'])
        d1_rep = term_encoder(bow_reps['d1']['input_ids'])
        d2_rep = term_encoder(bow_reps['d2']['input_ids'])
        get_doc_score = ScoringLayer(bert_params.hidden_size)
        s1, ex_tf1 = get_doc_score([q_rep, d1_rep, bow_reps['q'], bow_reps['d1']])
        s2, ex_tf2 = get_doc_score([q_rep, d2_rep, bow_reps['q'], bow_reps['d2']])

        loss_cont = tf.maximum(1 - (s1 - s2), 0)
        loss_tf1 = get_tf_loss(ex_tf1)
        loss_tf2 = get_tf_loss(ex_tf2)
        loss = loss_cont + loss_tf1 + loss_tf2
        output = [(s1, s2), loss]
        model = keras.Model(inputs=inputs, outputs=output, name="bow_translation_table")
        self.loss = loss
        self.model: keras.Model = model
        self.bert_cls = term_encoder.bert_cls


class TranslationTableQTW2:
    def __init__(self, bert_params, config: InputShapeConfigTT):
        q_term_encoder = TermEncoder(bert_params)
        d_term_encoder = TermEncoder(bert_params)

        window_len = config.max_subword_per_word
        num_window = config.max_terms
        bow_reps, inputs = define_inputs(num_window, window_len)

        batch_size, _, _ = get_shape_list2(bow_reps['q']['input_ids'])
        q_rep = q_term_encoder(bow_reps['q']['input_ids'])
        d1_rep = d_term_encoder(bow_reps['d1']['input_ids'])
        d2_rep = d_term_encoder(bow_reps['d2']['input_ids'])
        get_doc_score = ScoringLayer(bert_params.hidden_size)
        s1, ex_tf1  = get_doc_score([q_rep, d1_rep, bow_reps['q'], bow_reps['d1']])
        s2, ex_tf2  = get_doc_score([q_rep, d2_rep, bow_reps['q'], bow_reps['d2']])
        loss_cont = tf.maximum(1 - (s1 - s2), 0)
        loss_tf1 = get_tf_loss(ex_tf1)
        loss_tf2 = get_tf_loss(ex_tf2)
        loss = loss_cont + loss_tf1 + loss_tf2
        output = [(s1, s2), loss]
        model = keras.Model(inputs=inputs, outputs=output, name="bow_translation_table")
        self.loss = loss
        self.model: keras.Model = model
        self.bert_cls_list = [q_term_encoder.bert_cls, d_term_encoder.bert_cls]


class TranslationTable3:
    def __init__(self, bert_params, config: InputShapeConfigTT):
        q_term_encoder = TermEncoder(bert_params)
        d_term_encoder = TermEncoder(bert_params)

        window_len = config.max_subword_per_word
        num_window = config.max_terms
        bow_reps, inputs = define_inputs(num_window, window_len)

        batch_size, _, _ = get_shape_list2(bow_reps['q']['input_ids'])
        q_rep = q_term_encoder(bow_reps['q']['input_ids'])
        d1_rep = d_term_encoder(bow_reps['d1']['input_ids'])
        d2_rep = d_term_encoder(bow_reps['d2']['input_ids'])
        get_doc_score = ScoringLayer2(bert_params.hidden_size)
        s1, ex_tf1  = get_doc_score([q_rep, d1_rep, bow_reps['q'], bow_reps['d1']])
        s2, ex_tf2  = get_doc_score([q_rep, d2_rep, bow_reps['q'], bow_reps['d2']])

        margin = 1
        loss_cont = tf.maximum(margin - (s1 - s2), 0)
        loss_tf1 = get_tf_loss(ex_tf1)
        loss_tf2 = get_tf_loss(ex_tf2)
        verbosity_loss = loss_tf1 + loss_tf2
        loss = loss_cont + verbosity_loss
        output = [(s1, s2), loss, verbosity_loss]
        model = keras.Model(inputs=inputs, outputs=output, name="bow_translation_table")
        self.loss = loss
        self.verbosity_loss = verbosity_loss
        self.model: keras.Model = model
        self.bert_cls_list = [q_term_encoder.bert_cls, d_term_encoder.bert_cls]


class TranslationTableInference:
    def __init__(self, bert_params, config: InputShapeConfigTT):
        term_encoder = TermEncoder(bert_params)
        window_len = config.max_subword_per_word
        num_window = config.max_terms
        role_list = ["q", "d"]

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
        q_rep = term_encoder(bow_reps['q']['input_ids'])
        d1_rep = term_encoder(bow_reps['d']['input_ids'])

        get_doc_score = ScoringLayer(bert_params.hidden_size)
        s1, d_expanded_tf = get_doc_score([q_rep, d1_rep, bow_reps['q'], bow_reps['d1']])
        output = [s1]
        model = keras.Model(inputs=inputs, outputs=output, name="bow_translation_table")
        self.model: keras.Model = model
        self.bert_cls = term_encoder.bert_cls


class TranslationTableInferenceQTW:
    def __init__(self, bert_params, config: InputShapeConfigTT, term_encoder):
        window_len = config.max_subword_per_word
        num_window = config.max_terms
        role_list = ["q", "d"]

        inputs = []
        bow_reps = {}
        for role in role_list:
            input_ids = keras.layers.Input(shape=(num_window * window_len,), dtype='int32', name=f"{role}_input_ids")
            input_ids_stacked = tf.reshape(
                input_ids, [-1, num_window, window_len],
                name=f"{role}_input_ids_stacked")
            tfs = keras.layers.Input(shape=(num_window,), dtype='int32', name=f"{role}_tfs")
            qtw = keras.layers.Input(shape=(num_window,), dtype='float32', name=f"{role}_qtw")
            bow_reps[role] = {
                'input_ids': input_ids_stacked,
                'tfs': tfs,
                'qtw': qtw
            }
            inputs.append(input_ids)
            inputs.append(tfs)
            inputs.append(qtw)

        batch_size, _, _ = get_shape_list2(bow_reps['q']['input_ids'])
        q_rep = term_encoder(bow_reps['q']['input_ids'])
        d1_rep = term_encoder(bow_reps['d']['input_ids'])

        get_doc_score = ScoringLayer(bert_params.hidden_size)
        score, d_expanded_tf = get_doc_score([q_rep, d1_rep, bow_reps['q'], bow_reps['d']])
        output = [score]
        model = keras.Model(inputs=inputs, outputs=output, name="bow_translation_table")
        self.model: keras.Model = model
        self.bert_cls = term_encoder.bert_cls

