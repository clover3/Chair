import math
from typing import NamedTuple

import tensorflow as tf

from models.transformer import bert_common_v2 as bc
from models.transformer.bert_common_v2 import create_initializer, get_shape_list, get_activation, reshape_to_matrix, \
    dropout, create_attention_mask_from_input_mask, get_shape_list2, reshape_from_matrix


class MyLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(kwargs)


def embedding_postprocessor(input_tensor, token_type_table, full_position_embeddings,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            use_position_embeddings=True,
                            max_position_embeddings=512,
                            ):
    input_shape = get_shape_list2(input_tensor)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]
    output = input_tensor

    if use_token_type:
        if token_type_ids is None:
            raise ValueError("`token_type_ids` must be specified if"
                             "`use_token_type` is True.")
        # This vocab will be small so we always do one-hot here, since it is always
        # faster for a small vocabulary.
        flat_token_type_ids = tf.reshape(token_type_ids, [-1])
        one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
        token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
        token_type_embeddings = tf.reshape(token_type_embeddings,
                                           [batch_size, seq_length, width])
        output += token_type_embeddings

    if use_position_embeddings:
        assert_op = tf.compat.v1.assert_less_equal(seq_length, max_position_embeddings)
        with tf.control_dependencies([assert_op]):
            # Since the position embedding table is a learned variable, we create it
            # using a (long) sequence length `max_position_embeddings`. The actual
            # sequence length might be shorter than this, for faster training of
            # tasks that do not have long sequences.
            #
            # So `full_position_embeddings` is effectively an embedding table
            # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
            # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
            # perform a slice.
            position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                           [seq_length, -1])
            num_dims = len(output.shape.as_list())

            # Only the last two dimensions are relevant (`seq_length` and `width`), so
            # we broadcast among the first dimensions, which is typically just
            # the batch size.
            position_broadcast_shape = []
            for _ in range(num_dims - 2):
                position_broadcast_shape.append(1)
            position_broadcast_shape.extend([seq_length, width])
            position_embeddings = tf.reshape(position_embeddings, position_broadcast_shape)
            output += position_embeddings

    return output


class Embedding(MyLayer):
    def __init__(self, config, use_one_hot_embeddings):
        super(Embedding, self).__init__()
        self.config = config
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.embedding_output = None
        self.embedding_table = None
        self.config = config
        initializer = bc.create_initializer(config.initializer_range)
        self.embedding_table = tf.compat.v1.get_variable(
            name="word_embeddings",
            shape=[config.vocab_size, config.hidden_size],
            initializer=initializer)
        self.token_type_table = tf.compat.v1.get_variable(
            name="token_type_embeddings",
            shape=[config.type_vocab_size, config.hidden_size],
            initializer=initializer)
        self.full_position_embeddings = tf.compat.v1.get_variable(
            name="position_embeddings",
            shape=[config.max_position_embeddings, config.hidden_size],
            initializer=initializer)

        # Perform embedding lookup on the word ids.
    def call(self, inputs, **kwargs):
        input_ids, segment_ids = inputs
        config = self.config
        (self.embedding_output, self.embedding_table) = bc.embedding_lookup2(
            input_ids=input_ids,
            embedding_table=self.embedding_table,
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            use_one_hot_embeddings=self.use_one_hot_embeddings)

        self.location_less_embedding = self.embedding_output

        self.embedding_output = embedding_postprocessor(
            input_tensor=self.embedding_output,
            token_type_table=self.token_type_table,
            full_position_embeddings=self.full_position_embeddings,
            use_token_type=True,
            token_type_ids=segment_ids,
            token_type_vocab_size=config.type_vocab_size,
            use_position_embeddings=True,
            max_position_embeddings=config.max_position_embeddings)
        return self.embedding_output


def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                         seq_length, width):
    output_tensor = tf.reshape(
        input_tensor, [batch_size, seq_length, num_attention_heads, width])

    output_tensor = tf.transpose(a=output_tensor, perm=[0, 2, 1, 3])
    return output_tensor


class ProjectionLayer(MyLayer):
    def __init__(self, name, config, activation=None):
        super(ProjectionLayer, self).__init__()
        self.num_attention_heads = config.num_attention_heads
        self.size_per_head = config.size_per_head
        initializer = create_initializer(config.initializer_range)
        self.layer = tf.keras.layers.Dense(
            config.num_attention_heads * self.size_per_head,
            activation=activation,
            name=name,
            kernel_initializer=initializer)

    def call(self, input_tensor, **kwargs):
        v = self.layer(input_tensor)
        return v


class ProjectionDropoutLayer(MyLayer):
    def __init__(self, name, config, activation=None):
        super(ProjectionDropoutLayer, self).__init__()
        self.num_attention_heads = config.num_attention_heads
        self.size_per_head = config.size_per_head
        self.hidden_dropout_prob = config.hidden_dropout_prob
        initializer = create_initializer(config.initializer_range)
        self.layer = tf.keras.layers.Dense(
            config.num_attention_heads * self.size_per_head,
            activation=activation,
            name=name,
            kernel_initializer=initializer)

    def call(self, input_tensor, **kwargs):
        v = self.layer(input_tensor)
        # v = dropout(v, self.hidden_dropout_prob)
        return v


class AttentionWeightLayer(MyLayer):
    def __init__(self, config):
        super(AttentionWeightLayer, self).__init__()
        self.num_attention_heads = config.num_attention_heads
        self.size_per_head = config.size_per_head

    def call(self, inputs, **kwargs):
        query_vector, key_vector, batch_size, from_seq_length, to_seq_length = inputs
        query_layer = transpose_for_scores(query_vector,
                                           batch_size,
                                           self.num_attention_heads,
                                           from_seq_length,
                                           self.size_per_head)
        key_layer = transpose_for_scores(key_vector,
                                         batch_size,
                                         self.num_attention_heads,
                                         to_seq_length,
                                         self.size_per_head)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = tf.multiply(attention_scores,
                                       1.0 / math.sqrt(float(self.size_per_head)))
        return attention_scores


class ContextLayer(MyLayer):
    def __init__(self, config):
        super(ContextLayer, self).__init__()
        self.num_attention_heads = config.num_attention_heads
        self.size_per_head = config.size_per_head

    def call(self, inputs, **kwargs):
        value_vector, attention_probs = inputs
        weight_shape = get_shape_list2(attention_probs)
        batch_size = weight_shape[0]
        from_seq_length = weight_shape[2]
        to_seq_length = weight_shape[3]

        value_vector = tf.reshape(
            value_vector,
            [batch_size,
             to_seq_length,
             self.num_attention_heads,
             self.size_per_head], name="value_reshape")
        value_vector = tf.transpose(a=value_vector, perm=[0, 2, 1, 3])

        context_vector = tf.matmul(attention_probs, value_vector)
        # `context_layer` = [B, F, N, H]
        context_vector = tf.transpose(a=context_vector, perm=[0, 2, 1, 3])
        context_vector = tf.reshape(
            context_vector,
            [batch_size * from_seq_length,
             self.num_attention_heads * self.size_per_head])

        return context_vector


class IntermediateLayer(MyLayer):
    def __init__(self, name, config):
        super(IntermediateLayer, self).__init__()
        intermediate_act_fn = get_activation(config.hidden_act)
        self.layer = tf.keras.layers.Dense(config.intermediate_size,
                                           activation=intermediate_act_fn,
                                           kernel_initializer=create_initializer(config.initializer_range),
                                           name=name
                                           )

    def call(self, inputs, **kwargs):
        return self.layer(inputs)


class PerLayer(NamedTuple):
    query: ProjectionLayer
    key: ProjectionLayer
    value: ProjectionLayer
    attn_weight: AttentionWeightLayer
    context: ContextLayer
    attn_output: ProjectionLayer
    attention_layer_norm: tf.keras.layers.LayerNormalization
    intermediate: IntermediateLayer
    output: ProjectionLayer
    output_layer_norm: tf.keras.layers.LayerNormalization


def apply_attention_mask(attention_scores, attention_mask):
    # `attention_mask` = [B, 1, F, T]
    attention_mask = tf.expand_dims(attention_mask, axis=[1])
    adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
    attention_scores += adder
    return attention_scores



class ContribLayerNormalization(tf.keras.layers.Layer):
    """
    Layer normalization layer from arXiv:1607.06450.
        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
        See: https://github.com/CyberZHG/keras-layer-normalization
        See: tf.contrib.layers.layer_norm
    """

    def __init__(self, name):
        super(ContribLayerNormalization, self).__init__(name=name)
        self.epsilon = 1e-12

    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        self.gamma = None
        self.beta  = None
        self.supports_masking = True

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)
        self.gamma = self.add_weight(name="gamma", shape=input_shape[-1:],
                                     initializer=tf.keras.initializers.Ones(), trainable=True)
        self.beta  = self.add_weight(name="beta", shape=input_shape[-1:],
                                     initializer=tf.keras.initializers.Zeros(), trainable=True)
        super(ContribLayerNormalization, self).build(input_shape)

    def call(self, inputs, **kwargs):                               # pragma: no cover
        x = inputs
        if tf.__version__.startswith("2."):
            mean, var = tf.nn.moments(x, axes=-1, keepdims=True)
        else:
            mean, var = tf.nn.moments(x, axes=-1, keep_dims=True)
        inv = self.gamma * tf.math.rsqrt(var + self.epsilon)
        res = x * tf.cast(inv, x.dtype) + tf.cast(self.beta - mean * inv, x.dtype)

        return res


def get_layer_norm(name):
    # return ContribLayerNormalization(name)
    eps = 1e-12
    return tf.keras.layers.LayerNormalization(epsilon=eps, axis=-1, name=name)


class BertLayer(MyLayer):
    def __init__(self, config, use_one_hot_embeddings, do_return_all_layers):
        super(BertLayer, self).__init__()
        config.size_per_head = int(config.hidden_size / config.num_attention_heads)
        self.config = config
        self.do_return_all_layers = do_return_all_layers
        self.use_one_hot_embeddings = use_one_hot_embeddings
        with tf.compat.v1.variable_scope("embeddings") as name_scope:
            self.embedding_layer = Embedding(config, use_one_hot_embeddings)
            full_name = name_scope.name + "/LayerNorm"
            self.embedding_layer_norm = get_layer_norm(full_name)
        attn_comp_names = ["query", "key", "value"]
        self.layers = []
        self.layer_d_list = []
        with tf.compat.v1.variable_scope("encoder"):
            for layer_idx in range(config.num_hidden_layers):
                layer_d = {}
                with tf.compat.v1.variable_scope("layer_%d" % layer_idx):
                    with tf.compat.v1.variable_scope("attention"):
                        with tf.compat.v1.variable_scope("self") as name_scope:
                            for name in attn_comp_names:
                                full_name = name_scope.name + "/" + name
                                layer = ProjectionLayer(full_name, config)
                                layer_d[layer_idx, name] = layer
                            layer_d[layer_idx, "attn_weight"] = AttentionWeightLayer(config)
                            layer_d[layer_idx, "context"] = ContextLayer(config)

                        with tf.compat.v1.variable_scope("output") as name_scope:
                            full_name = name_scope.name + "/dense"
                            layer_d[layer_idx, "attention_output"] = ProjectionDropoutLayer(full_name, config)
                            full_name = name_scope.name + "/LayerNorm"
                            layer_d[layer_idx, "attention_layer_norm"]\
                                = get_layer_norm(name=full_name)
                    with tf.compat.v1.variable_scope("intermediate") as name_scope:
                        full_name = name_scope.name + "/" + "dense"
                        layer_d[layer_idx, "intermediate"] = IntermediateLayer(full_name, config)
                    with tf.compat.v1.variable_scope("output") as name_scope:
                        full_name = name_scope.name + "/dense"
                        layer = ProjectionDropoutLayer(full_name, config)
                        layer_d[layer_idx, "output"] = layer
                        full_name = name_scope.name + "/LayerNorm"
                        layer_norm = get_layer_norm(name=full_name)
                        layer_d[layer_idx, "output_layer_norm"] = layer_norm

                layer_comps = PerLayer(layer_d[layer_idx, "query"],
                             layer_d[layer_idx, "key"],
                             layer_d[layer_idx, "value"],
                             layer_d[layer_idx, "attn_weight"],
                             layer_d[layer_idx, "context"],
                             layer_d[layer_idx, "attention_output"],
                             layer_d[layer_idx, "attention_layer_norm"],
                             layer_d[layer_idx, "intermediate"],
                             layer_d[layer_idx, "output"],
                             layer_d[layer_idx, "output_layer_norm"],
                             )
                self.layers.append(layer_comps)
                self.layer_d_list.append(layer_d)

    def call(self, inputs, **kwargs):
        input_ids, input_mask, segment_ids = inputs
        with tf.compat.v1.variable_scope("embeddings") as name_scope:
            embedding_output = self.embedding_layer((input_ids, segment_ids))
            embedding_output = self.embedding_layer_norm(embedding_output)
            embedding_output = dropout(embedding_output, self.config.hidden_dropout_prob)
            self.embedding_output = embedding_output

        with tf.compat.v1.variable_scope("encoder"):
            input_shape = get_shape_list(embedding_output, expected_rank=3)
            batch_size = input_shape[0]
            seq_length = input_shape[1]
            prev_output = reshape_to_matrix(embedding_output)
            attention_mask = create_attention_mask_from_input_mask(
                input_ids, input_mask)

            all_layer_outputs = []
            for layer_idx in range(self.config.num_hidden_layers):
                layer = self.layers[layer_idx]
                layer_input = prev_output
                query = layer.query.call(layer_input)
                key = layer.key.call(layer_input)
                value = layer.value.call(layer_input)

                inputs = query, key, batch_size, seq_length, seq_length
                attention_scores = layer.attn_weight.call(inputs)
                attention_scores = apply_attention_mask(attention_scores, attention_mask)
                attention_probs = tf.nn.softmax(attention_scores)
                context_v = layer.context.call((value, attention_probs))
                attention_output = layer.attn_output.call(context_v)
                attention_output = layer.attention_layer_norm(attention_output + layer_input)

                intermediate_output = layer.intermediate.call(attention_output)
                layer_output = layer.output.call(intermediate_output)
                layer_output = layer.output_layer_norm(layer_output + attention_output)
                prev_output = layer_output
                all_layer_outputs.append(layer_output)

        return self.summarize_output(all_layer_outputs, input_shape)

    def summarize_output(self, all_layer_outputs, input_shape):
        if self.do_return_all_layers:
            final_outputs = []
            for layer_output in all_layer_outputs:
                final_output = reshape_from_matrix(layer_output, input_shape)
                final_outputs.append(final_output)
            return final_outputs
        else:
            final_output = reshape_from_matrix(all_layer_outputs[-1], input_shape)
            return final_output


class BertClassifierLayer(MyLayer):
    def __init__(self, config, use_one_hot_embeddings, num_classes, is_training=False):
        super(BertClassifierLayer, self).__init__()
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        self.num_classes = num_classes
        with tf.compat.v1.variable_scope("bert"):
            self.bert_layer = BertLayer(config, use_one_hot_embeddings, True)
            with tf.compat.v1.variable_scope("pooler") as name_scope:
                self.pooler = tf.keras.layers.Dense(config.hidden_size,
                                                    activation=tf.keras.activations.tanh,
                                                    kernel_initializer=create_initializer(config.initializer_range),
                                                    name=name_scope.name + "/dense"
                                                    )

        self.output_weights = tf.compat.v1.get_variable(
            "output_weights", [num_classes, config.hidden_size],
            initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02)
        )
        self.output_bias = tf.compat.v1.get_variable(
            "output_bias", [num_classes],
            initializer=tf.compat.v1.zeros_initializer()
        )
        self.is_training = is_training

    def call(self, inputs, **kwargs):
        with tf.compat.v1.variable_scope("bert"):
            sequence_output = self.bert_layer.call(inputs)
            self.sequence_output = sequence_output
            last_layer = sequence_output[-1]
            first_token_tensor = tf.squeeze(last_layer[:, 0:1, :], axis=1)
            pooled = self.pooler(first_token_tensor)
        self.pooled_output = pooled
        if self.is_training:
            pooled = dropout(pooled, 0.1)

        text_rep = pooled
        logits = tf.matmul(text_rep, self.output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, self.output_bias)
        return logits


def define_bert_keras_inputs(max_seq_len):
    l_input_ids = tf.keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
    l_input_mask = tf.keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_mask")
    l_token_type_ids = tf.keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="segment_ids")
    inputs = (l_input_ids, l_input_mask, l_token_type_ids)
    return inputs

