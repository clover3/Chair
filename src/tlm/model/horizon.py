import tensorflow as tf

from models.transformer import bert_common_v2 as bc
from models.transformer.bert_common_v2 import get_shape_list2, create_attention_mask_from_input_mask2
from tlm.model import base
from tlm.model.base import mimic_pooling
from tlm.model.units import Embedding2


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
        from_tensor, to_tensor, attention_mask = inputs
        batch_size, from_seq_len, _ = get_shape_list2(from_tensor)
        batch_size, to_seq_len, _ = get_shape_list2(to_tensor)

        attention_output = bc.attention_layer2(
            from_tensor=from_tensor,
            to_tensor=to_tensor,
            query_ff=self.sub_layers['query'],
            key_ff=self.sub_layers['key'],
            value_ff=self.sub_layers['value'],
            attention_mask=attention_mask,
            num_attention_heads=self.num_attention_heads,
            size_per_head=self.attention_head_size,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            do_return_2d_tensor=False,
            batch_size=batch_size,
            from_seq_length=from_seq_len,
            to_seq_length=to_seq_len)

        attention_output = self.sub_layers['output'](attention_output)
        attention_output = bc.dropout(attention_output, self.hidden_dropout_prob)
        attention_output = bc.layer_norm(attention_output + from_tensor)
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
        return bc.layer_norm(layer_output + inputs)


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

    def __call__(self, from_tensor, to_tensor, attention_mask):
        e = from_tensor, to_tensor, attention_mask
        with tf.compat.v1.variable_scope("attention"):
            attention_output = self.attention_unit(e)
        with tf.compat.v1.variable_scope("feed_forward"):
            layer_output = self.residual_ff(attention_output)
        return layer_output


class HorizonForwardLayer(tf.keras.layers.Layer):
    # It does not decide how many tokens we use
    def __init__(self, config):
        super(HorizonForwardLayer, self).__init__()
        self.config = config
        num_columns = config.num_columns
        self.column_list = []
        for tower_idx in range(num_columns):
            column = ForwardColumn(config)
            self.column_list.append(column)

    def __call__(self, tensor_list, to_tensor_mask):
        to_tensor = tf.concat(tensor_list, axis=1) # [batch, seq_len + num_col, hidden_dim]
        batch_size, to_seq_length, _ = get_shape_list2(to_tensor)

        out_tensor_list = []
        for column_idx, column in enumerate(self.column_list):
            from_tensor = tensor_list[column_idx]
            _, from_seq_length, _ = get_shape_list2(to_tensor)
            attention_mask = create_attention_mask_from_input_mask2(from_tensor, to_tensor_mask)

            with tf.compat.v1.variable_scope("Column_{}".format(column_idx)):
                out_tensor = column(from_tensor, to_tensor, attention_mask)
            out_tensor_list.append(out_tensor)
        return out_tensor_list, to_tensor_mask


class HorizontalAlpha(base.BertModelInterface):
    def __init__(self, config, is_training, use_one_hot_embeddings):
        super(HorizontalAlpha, self).__init__()
        if not is_training:
            config.set_attrib("hidden_dropout_prob", 0.0)
            config.set_attrib("attention_probs_dropout_prob", 0.0)

        initializer = bc.create_initializer(config.initializer_range)
        self.embedding_layer = Embedding2()
        self.embedding_projector = bc.dense(config.hidden_size, initializer)
        self.forward_layer = HorizonForwardLayer(config)
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
        t = tf.tile(tf.expand_dims(self.column_mask, 0), [batch_size, 1])
        t = tf.concat([input_mask, t], axis=1)
        return t

    def embedding_projection(self, input_tensor):
        with tf.compat.v1.variable_scope("embedding_projection", reuse=True):
            return self.embedding_projector(input_tensor)

    def call(self, input_ids, input_mask, segment_ids):
        input_ids = input_ids
        input_mask = input_mask
        segment_ids = segment_ids

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
        to_tensor_mask = self.get_to_tensor_mask(batch_size, input_mask)
        for layer_no in range(self.num_layers):
            with tf.compat.v1.variable_scope("layer", reuse=layer_no > 0):
                tensor_list, to_tensor_mask = self.forward_layer(tensor_list, to_tensor_mask)
                self.all_raw_layers.append(tensor_list)
                self.all_main_layers.append(tensor_list[0])

        self.embedding_table = self.embedding_layer.embedding_table
        self.sequence_output = self.all_main_layers[-1]
        self.pooled_output = mimic_pooling(self.sequence_output, self.config.hidden_size, self.config.initializer_range)
        return self.sequence_output