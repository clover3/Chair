from typing import List

import math
import tensorflow as tf

from models.transformer import bert_common_v2 as bc
from models.transformer.bert_common_v2 import get_shape_list2, create_attention_mask_from_size
from tlm.model import base
from tlm.model.base import mimic_pooling
from tlm.model.units import Embedding2

debug_mode = False

def init_query_key_value(num_attention_heads, attention_head_size, initializer):
    query_layer = tf.keras.layers.Dense(
        num_attention_heads * attention_head_size,
        activation=None,
        name="query",
        kernel_initializer=initializer)

    key_layer = tf.keras.layers.Dense(
        num_attention_heads * attention_head_size,
        activation=None,
        name="key",
        kernel_initializer=initializer)
    value_layer = tf.keras.layers.Dense(
        num_attention_heads * attention_head_size,
        activation=None,
        name="value",
        kernel_initializer=initializer)
    return query_layer, key_layer, value_layer


class Tensor2D:
    def __init__(self, tensor_3d):
        self.batch_size, self.seq_length, self.hidden_dims = get_shape_list2(tensor_3d)
        self.matrix = tf.reshape(tensor_3d, [-1, self.hidden_dims])

    def get_3d(self):
        return tf.reshape(self.matrix, [self.batch_size, self.seq_length, -1])


def attention_layer(from_tensor: Tensor2D,
                    to_tensor_list: List[Tensor2D],
                    query_ff,
                    key_ff,
                    value_ff,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    attention_probs_dropout_prob=0.0):

    def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                                                     seq_length, width):
        output_tensor = tf.reshape(
                input_tensor, [batch_size, seq_length, num_attention_heads, width], name="reshape_transpose_for_scores")

        output_tensor = tf.transpose(a=output_tensor, perm=[0, 2, 1, 3])
        return output_tensor

    from_shape = get_shape_list2(from_tensor.matrix)
    for to_tensor in to_tensor_list:
        to_shape = get_shape_list2(to_tensor.matrix)
        if len(from_shape) != len(to_shape):
            raise ValueError(
                    "The rank of `from_tensor` must match the rank of `to_tensor`.")

    # `query_layer` = [B*F, N*H]
    query_layer = query_ff(from_tensor.matrix)
    # `query_layer` = [B, N, F, H]
    query_layer = transpose_for_scores(query_layer, from_tensor.batch_size, num_attention_heads,
                                       from_tensor.seq_length, size_per_head)

    key_layer_list = []
    value_layer_list = []
    for to_tensor in to_tensor_list:
        # `key_layer` = [B*T, N*H]
        key_layer = key_ff(to_tensor.matrix)

        # `key_layer` = [B, N, T, H]
        key_layer = transpose_for_scores(key_layer, to_tensor.batch_size,
                                         num_attention_heads, to_tensor.seq_length, size_per_head)

        key_layer_list.append(key_layer)
        # `value_layer` = [B*T, N*H]
        value_layer = value_ff(to_tensor.matrix)

        # `value_layer` = [B, T, N, H]
        value_layer = tf.reshape(value_layer,
                                 [to_tensor.batch_size, to_tensor.seq_length, num_attention_heads, size_per_head],
                                 name="value_reshape")

        # `value_layer` = [B, N, T, H]
        value_layer = tf.transpose(a=value_layer, perm=[0, 2, 1, 3])
        value_layer_list.append(value_layer)

    key_layer_all = tf.concat(key_layer_list, axis=2)
    value_layer_all = tf.concat(value_layer_list, axis=2)

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    # `attention_scores` = [B, N, F, T]
    attention_scores = tf.matmul(query_layer, key_layer_all, transpose_b=True)
    attention_scores = tf.multiply(attention_scores, 1.0 / math.sqrt(float(size_per_head)))

    if attention_mask is not None:
        # `attention_mask` = [B, 1, F, T]
        attention_mask = tf.expand_dims(attention_mask, axis=[1])

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        attention_scores += adder

    # Normalize the attention scores to probabilities.
    # `attention_probs` = [B, N, F, T]
    attention_probs = tf.nn.softmax(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    # TODO restore this
    # attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

    # `context_layer` = [B, N, F, H]
    context_layer = tf.matmul(attention_probs, value_layer_all)

    # `context_layer` = [B, F, N, H]
    context_layer = tf.transpose(a=context_layer, perm=[0, 2, 1, 3])

    # `context_layer` = [B*F, N*V]
    context_layer = tf.reshape(
            context_layer,
            [from_tensor.batch_size * from_tensor.seq_length, num_attention_heads * size_per_head])

    return context_layer


class AttentionUnit(tf.keras.layers.Layer):
    def __init__(self, num_attention_heads, attention_head_size, hidden_size,
                 hidden_dropout_prob,
                 attention_probs_dropout_prob, initializer):
        super(AttentionUnit, self).__init__()
        query_layer, key_layer, value_layer \
            = init_query_key_value(num_attention_heads, attention_head_size, initializer)
        output_layer = tf.keras.layers.Dense(hidden_size, kernel_initializer=initializer)
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = attention_head_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob

        self.sub_layers = {
            'query': query_layer,
            'key': key_layer,
            'value': value_layer,
            'output': output_layer
        }

    def __call__(self,  inputs):
        from_tensor, to_tensor_list, attention_mask = inputs

        attention_output = attention_layer(
            from_tensor=from_tensor,
            to_tensor_list=to_tensor_list,
            query_ff=self.sub_layers['query'],
            key_ff=self.sub_layers['key'],
            value_ff=self.sub_layers['value'],
            attention_mask=attention_mask,
            num_attention_heads=self.num_attention_heads,
            size_per_head=self.attention_head_size,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
        )

        attention_output = self.sub_layers['output'](attention_output)
        attention_output = bc.dropout(attention_output, self.hidden_dropout_prob)
        attention_output = bc.layer_norm(attention_output + from_tensor.matrix)
        return attention_output


class ResidualFeedforward(tf.keras.layers.Layer):
    def __init__(self, hidden_size, intermediate_size, hidden_act, hidden_dropout_prob, initializer):
        super(ResidualFeedforward, self).__init__()
        self.intermediate_ff = bc.dense(intermediate_size, initializer,
                                        activation=bc.get_activation(hidden_act))
        self.hidden_dropout_prob = hidden_dropout_prob
        self.output_ff = bc.dense(hidden_size, initializer)

    def __call__(self, inputs):
        intermediate_output = self.intermediate_ff(inputs)
        layer_output = self.output_ff(intermediate_output)
        layer_output = bc.dropout(layer_output, self.hidden_dropout_prob)
        layer_output = bc.layer_norm(layer_output + inputs)
        return layer_output


class ForwardColumn(tf.keras.layers.Layer):
    def __init__(self, config):
        super(ForwardColumn, self).__init__()
        hidden_size = config.hidden_size
        initializer = bc.create_initializer(config.initializer_range)

        attention_head_size = int(hidden_size / config.num_attention_heads)
        self.attention_head_size = attention_head_size
        num_attention_heads = config.num_attention_heads
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob
        self.hidden_dropout_prob = config.hidden_dropout_prob

        self.attention_unit = AttentionUnit(num_attention_heads,
                                       attention_head_size,
                                       hidden_size,
                                       config.hidden_dropout_prob,
                                       config.attention_probs_dropout_prob,
                                       initializer)
        self.residual_ff = ResidualFeedforward(hidden_size,
                                      config.intermediate_size,
                                      config.hidden_act,
                                      config.hidden_dropout_prob,
                                      initializer)
        self.attention_mask = None

    def __call__(self,
                 from_tensor: Tensor2D,
                 to_tensor_list: List[Tensor2D]
                 ):
        e = from_tensor, to_tensor_list, self.attention_mask
        if debug_mode:
            with tf.compat.v1.variable_scope("attention"):
                attention_output = self.attention_unit(e)
        else:
                attention_output = self.attention_unit(e)
        if debug_mode:
            with tf.compat.v1.variable_scope("feed_forward"):
                layer_output = self.residual_ff(attention_output)
        else:
            layer_output = self.residual_ff(attention_output)
        from_tensor.matrix = layer_output
        return from_tensor

    def check_attention_mask(self, from_tensor, to_tensor_mask):
        if self.attention_mask is None:
            self.attention_mask = create_attention_mask_from_size(from_tensor.batch_size,
                                                             from_tensor.seq_length,
                                                             to_tensor_mask)


class HorizontalAlpha(base.BertModelInterface):
    def __init__(self, config, is_training, use_one_hot_embeddings):
        super(HorizontalAlpha, self).__init__()
        if not is_training:
            config.set_attrib("hidden_dropout_prob", 0.0)
            config.set_attrib("attention_probs_dropout_prob", 0.0)

        initializer = bc.create_initializer(config.initializer_range)
        self.embedding_layer = Embedding2()
        self.embedding_projector = bc.dense(config.hidden_size, initializer)
        self.config = config
        num_columns = config.num_columns
        self.column_list = []
        for tower_idx in range(num_columns):
            column = ForwardColumn(config)
            self.column_list.append(column)

        self.num_layers = config.num_hidden_layers
        self.num_columns = config.num_columns
        self.num_column_tokens = config.num_column_tokens
        self.column_embedding_list = []
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.config = config
        column_mask = []
        for column_idx in range(1, self.num_columns):
            column_embedding = tf.Variable(lambda : initializer(shape=(self.num_column_tokens, config.hidden_size),
                                                       dtype=tf.float32),
                                           name="column_embedding_{}".format(column_idx))
            self.column_embedding_list.append(column_embedding)
            column_mask += [1] * self.num_column_tokens

        self.column_mask = tf.constant(column_mask)
        self.all_raw_layers = []
        self.all_main_layers = []
        self.sequence_output = None
        self.pooled_output = None

    def get_column_embeddings(self, batch_size):
        output = []
        for column_embedding in self.column_embedding_list:
            c_emb = tf.tile(tf.expand_dims(column_embedding, 0), [batch_size, 1, 1])
            output.append(c_emb)
        return output

    def get_to_tensor_mask(self, batch_size, input_mask):
        # [batch_size, seq_len + column_len]
        t = tf.tile(tf.expand_dims(self.column_mask, 0), [batch_size, 1])
        t = tf.concat([input_mask, t], axis=1)
        return t

    def embedding_projection(self, input_tensor):
        if debug_mode:
            with tf.compat.v1.variable_scope("embedding_projection", reuse=True):
                return self.embedding_projector(input_tensor)
        else:
            return self.embedding_projector(input_tensor)

    def forward(self, tensor_list, to_tensor_mask):
        out_tensor_list = []
        for column_idx, column in enumerate(self.column_list):
            from_tensor = tensor_list[column_idx]
            column.check_attention_mask(from_tensor, to_tensor_mask)
            with tf.compat.v1.variable_scope("Column_{}".format(column_idx)):
                out_tensor = column(from_tensor, tensor_list)
            out_tensor_list.append(out_tensor)
        return out_tensor_list

    def call(self, input_ids, input_mask, segment_ids):
        n_added_tokens = self.num_column_tokens * self.num_columns
        input_ids = input_ids[:, :-n_added_tokens]
        input_mask = input_mask[:, :-n_added_tokens]
        segment_ids = segment_ids[:, :-n_added_tokens]
        input_tensor = self.embedding_layer.apply(input_ids, segment_ids,
                                                  self.config.initializer_range,
                                                  self.config.vocab_size,
                                                  self.config.embedding_size,
                                                  self.config.type_vocab_size,
                                                  self.config.max_position_embeddings,
                                                  self.config.hidden_dropout_prob,
                                                  self.use_one_hot_embeddings)
        self.embedding_output = input_tensor
        input_tensor = self.embedding_projector(input_tensor) # [ batch_size, seq_len, hidden_dim ]

        batch_size, _, _ = get_shape_list2(input_tensor)

        tensor_list = [input_tensor] + self.get_column_embeddings(batch_size)
        tensor_list = [Tensor2D(t) for t in tensor_list]
        to_tensor_mask = self.get_to_tensor_mask(batch_size, input_mask)
        for layer_no in range(self.num_layers):
            with tf.compat.v1.variable_scope("layer", reuse=layer_no > 0):
                tensor_list = self.forward(tensor_list, to_tensor_mask)
                self.all_raw_layers.append(tensor_list)
                self.all_main_layers.append(tensor_list[0])

        self.embedding_table = self.embedding_layer.embedding_table

        last_main_tensor = self.all_main_layers[-1]
        self.sequence_output = last_main_tensor.get_3d()
        self.sequence_output = tf.concat([self.sequence_output,
                                          tf.zeros([batch_size, n_added_tokens, self.config.hidden_size])],
                                         axis=1)
        self.pooled_output = mimic_pooling(self.sequence_output, self.config.hidden_size, self.config.initializer_range)
        return self.sequence_output