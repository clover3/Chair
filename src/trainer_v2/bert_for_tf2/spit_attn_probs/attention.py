# coding=utf-8
#
# created by kpe on 15.Mar.2019 at 12:52
#

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from bert.layer import Layer
# from tensorflow.python import keras
from tensorflow import keras
from tensorflow.python.keras import backend as K


# This takes attention mask as input parameter
class AttentionLayerSAP_QK(Layer):
    class Params(Layer.Params):
        num_heads         = None
        size_per_head     = None
        initializer_range = 0.02
        query_activation  = None
        key_activation    = None
        value_activation  = None
        attention_dropout = 0.1
        negative_infinity = -10000.0  # used for attention scores before softmax

    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        self.query_activation = self.params.query_activation
        self.key_activation   = self.params.key_activation

        self.query_layer = None
        self.key_layer   = None

        self.supports_masking = True

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        self.input_spec = keras.layers.InputSpec(shape=input_shape)

        dense_units = self.params.num_heads * self.params.size_per_head  # N*H
        #
        # B, F, T, N, H - batch, from_seq_len, to_seq_len, num_heads, size_per_head
        #
        self.query_layer = keras.layers.Dense(units=dense_units, activation=self.query_activation,
                                              kernel_initializer=self.create_initializer(),
                                              name="query")
        self.key_layer   = keras.layers.Dense(units=dense_units, activation=self.key_activation,
                                              kernel_initializer=self.create_initializer(),
                                              name="key")
        self.dropout_layer = keras.layers.Dropout(self.params.attention_dropout)

        super(AttentionLayerSAP_QK, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        from_shape = input_shape

        # from_shape         # [B, F, W]   [batch_size, from_seq_length, from_width]
        # input_mask_shape   # [B, F]

        output_shape = [from_shape[0], from_shape[1], self.params.num_heads * self.params.size_per_head]

        return output_shape  # [B, F, N*H]

    # noinspection PyUnusedLocal
    def call(self, inputs, attention_mask, training=None, **kwargs):
        from_tensor = inputs
        to_tensor   = inputs
        #  from_tensor shape - [batch_size, from_seq_length, from_width]
        input_shape  = tf.shape(input=from_tensor)
        batch_size, from_seq_len, from_width = input_shape[0], input_shape[1], input_shape[2]
        to_seq_len = from_seq_len

        # [B, F, N*H] -> [B, N, F, H]
        def transpose_for_scores(input_tensor, seq_len):
            output_shape = [batch_size, seq_len,
                            self.params.num_heads, self.params.size_per_head]
            output_tensor = K.reshape(input_tensor, output_shape)
            return tf.transpose(a=output_tensor, perm=[0, 2, 1, 3])  # [B,N,F,H]

        query = self.query_layer(from_tensor)  # [B,F, N*H] [batch_size, from_seq_len, N*H]
        key   = self.key_layer(to_tensor)      # [B,T, N*H]

        query = transpose_for_scores(query, from_seq_len)           # [B, N, F, H]
        key   = transpose_for_scores(key,   to_seq_len)             # [B, N, T, H]

        attention_scores = tf.matmul(query, key, transpose_b=True)  # [B, N, F, T]
        attention_scores = attention_scores / tf.sqrt(float(self.params.size_per_head))

        if attention_mask is not None:
            attention_mask = tf.expand_dims(attention_mask, axis=1)  # [B, 1, F, T]
            # {1, 0} -> {0.0, -inf}
            adder = (1.0 - tf.cast(attention_mask, tf.float32)) * self.params.negative_infinity
            attention_scores = tf.add(attention_scores, adder)  # adding to softmax -> its like removing them entirely

        # scores to probabilities
        self.attention_scores = attention_scores
        attention_probs = tf.nn.softmax(attention_scores)           # [B, N, F, T]
        self.attention_probs = attention_probs
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout_layer(attention_probs,
                                             training=training)    # [B, N, F, T]

        return attention_probs                                                            # [B, F, N*H]

    # noinspection PyUnusedLocal
    def compute_mask(self, inputs, mask=None):
        return mask   # [B, F]


class AttentionLayerSAP_V(Layer):
    class Params(Layer.Params):
        num_heads         = None
        size_per_head     = None
        initializer_range = 0.02
        query_activation  = None
        key_activation    = None
        value_activation  = None
        attention_dropout = 0.1
        negative_infinity = -10000.0  # used for attention scores before softmax

    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        self.value_activation = self.params.value_activation
        self.value_layer = None

        self.supports_masking = True

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        dense_units = self.params.num_heads * self.params.size_per_head  # N*H
        #
        # B, F, T, N, H - batch, from_seq_len, to_seq_len, num_heads, size_per_head
        #
        self.value_layer = keras.layers.Dense(units=dense_units, activation=self.value_activation,
                                              kernel_initializer=self.create_initializer(),
                                              name="value")
        self.dropout_layer = keras.layers.Dropout(self.params.attention_dropout)

        super(AttentionLayerSAP_V, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        from_shape = input_shape

        # from_shape         # [B, F, W]   [batch_size, from_seq_length, from_width]
        # input_mask_shape   # [B, F]

        output_shape = [from_shape[0], from_shape[1], self.params.num_heads * self.params.size_per_head]

        return output_shape  # [B, F, N*H]

    # noinspection PyUnusedLocal
    def call(self, inputs, training=None, **kwargs):
        from_tensor, attention_probs = inputs
        to_tensor = from_tensor
        #  from_tensor shape - [batch_size, from_seq_length, from_width]
        input_shape  = tf.shape(input=from_tensor)
        batch_size, from_seq_len, from_width = input_shape[0], input_shape[1], input_shape[2]
        to_seq_len = from_seq_len
        value = self.value_layer(to_tensor)    # [B,T, N*H]

        # [B,T,N,H]
        value = tf.reshape(value, [batch_size, to_seq_len,
                                   self.params.num_heads, self.params.size_per_head])
        # [B,N,T,H]
        value = tf.transpose(a=value, perm=[0, 2, 1, 3])                                # [B, N, T, H]

        # [B, N, F, T] * [B,N,T,H] -> [B, N, F, H]
        context_layer = tf.matmul(attention_probs, value)                               # [B, N, F, H]
        context_layer = tf.transpose(a=context_layer, perm=[0, 2, 1, 3])                # [B, F, N, H]

        # [B, N, F, T] * [B,N,T,H] -> [B, F, N * H]
        output_shape = [batch_size, from_seq_len,
                        self.params.num_heads * self.params.size_per_head]
        context_layer = tf.reshape(context_layer, output_shape)
        return context_layer                                                            # [B, F, N*H]

    # noinspection PyUnusedLocal
    def compute_mask(self, inputs, mask=None):
        return mask   # [B, F]


# This takes attention mask as input parameter
class AttentionLayerSAP(Layer):
    class Params(Layer.Params):
        num_heads         = None
        size_per_head     = None
        initializer_range = 0.02
        query_activation  = None
        key_activation    = None
        value_activation  = None
        attention_dropout = 0.1
        negative_infinity = -10000.0  # used for attention scores before softmax

    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        self.layer_b = None

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        self.input_spec = keras.layers.InputSpec(shape=input_shape)
        self.layer_a = AttentionLayerSAP_QK.from_params(self.params,
                                                        name="self",
                                                        )

        self.layer_b = AttentionLayerSAP_V.from_params(self.params,
                                                       name="self",
                                                       )


        super(AttentionLayerSAP, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        from_shape = input_shape

        # from_shape         # [B, F, W]   [batch_size, from_seq_length, from_width]
        # input_mask_shape   # [B, F]

        output_shape = [from_shape[0], from_shape[1], self.params.num_heads * self.params.size_per_head]
        return output_shape  # [B, F, N*H]

    # noinspection PyUnusedLocal
    def call(self, inputs, attention_mask, training=None, **kwargs):
        attention_probs = self.layer_a(inputs, attention_mask)
        context_layer = self.layer_b([inputs, attention_probs])
        return context_layer                                                            # [B, F, N*H]

    # noinspection PyUnusedLocal
    def compute_mask(self, inputs, mask=None):
        return mask   # [B, F]
