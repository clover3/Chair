import tensorflow as tf
from tensorflow import keras

from models.transformer.bert_common_v2 import get_shape_list2
from trainer_v2.bert_for_tf2 import BertModelLayer
from trainer_v2.custom_loop.modeling_common.bert_common import BERT_CLS


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


class TermVector(tf.keras.layers.Layer):
    def __init__(self, bert_params, role, trainable=False):
        super(TermVector, self).__init__()
        self.interaction_rep_size = bert_params.hidden_size

        def init_word_embedding(name):
            return keras.layers.Embedding(
                input_dim=bert_params.vocab_size,
                output_dim=bert_params.hidden_size,
                mask_zero=bert_params.mask_zero,
                name=name,
                trainable=trainable
            )
        self.embeddings_layer = init_word_embedding(f"{role}_term_embedding")
        self.dense = tf.keras.layers.Dense(self.interaction_rep_size,
                                            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                                                stddev=1e-3)
                                            )

    def call(self, inputs):
        input_ids = inputs  # [B, max_term, term_len]
        batch_size, num_terms, window_len = get_shape_list2(input_ids)
        input_ids_flat = tf.reshape(input_ids, [-1, window_len])  # [B * max_term, term_len]
        first_subword = input_ids_flat[:, 0]
        token_output = self.embeddings_layer(first_subword)
        item_rep = self.dense(token_output)
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