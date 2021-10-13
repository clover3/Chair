
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import tensorflow as tf

from models.transformer.bert import embedding_postprocessor, embedding_lookup, create_attention_mask_from_input_mask, \
    get_shape_list, get_activation, attention_layer, create_initializer, dropout, layer_norm


class BertMiddleIn(object):
    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 use_one_hot_embeddings=True,
                 embeddding_as_input=None,
                 middle_layer=0,
                 scope=None):
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        input_shape = get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.intermediate_size = config.intermediate_size
        self.intermediate_act_fn = get_activation(config.hidden_act)
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob
        self.initializer_range = config.initializer_range
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)

        all_layer_outputs = []

        with tf.variable_scope(scope, default_name="bert"):
            with tf.variable_scope("embeddings"):
                # Perform embedding lookup on the word ids.
                (self.embedding_output, self.embedding_table) = embedding_lookup(
                    input_ids=input_ids,
                    vocab_size=config.vocab_size,
                    embedding_size=config.hidden_size,
                    initializer_range=config.initializer_range,
                    word_embedding_name="word_embeddings",
                    use_one_hot_embeddings=use_one_hot_embeddings)

                # Add positional embeddings and token type embeddings, then layer
                # normalize and perform dropout.
                self.embedding_output = embedding_postprocessor(
                    input_tensor=self.embedding_output,
                    use_token_type=True,
                    token_type_ids=token_type_ids,
                    token_type_vocab_size=config.type_vocab_size,
                    token_type_embedding_name="token_type_embeddings",
                    use_position_embeddings=True,
                    position_embedding_name="position_embeddings",
                    initializer_range=config.initializer_range,
                    max_position_embeddings=config.max_position_embeddings,
                    dropout_prob=config.hidden_dropout_prob)
            with tf.variable_scope("encoder"):
                attention_mask = create_attention_mask_from_input_mask(
                    input_ids, input_mask)
                self.attention_mask = attention_mask
                prev_output = self.embedding_output
                for layer_idx in range(middle_layer):
                    with tf.variable_scope("layer_%d" % layer_idx):
                        layer_input = prev_output
                        layer_output = self.build_layer(layer_input, attention_mask)
                    prev_output = layer_output
                    if layer_idx != middle_layer - 1:
                        all_layer_outputs.append(layer_output)
                self.middle_output = prev_output, attention_mask

                if embeddding_as_input is not None:
                    print("Using embeddding_as_input")
                    encoded_embedding, attention_mask = embeddding_as_input
                else:
                    encoded_embedding = prev_output
                    attention_mask = attention_mask
                all_layer_outputs.append(encoded_embedding)
                prev_output = encoded_embedding
                for layer_idx in range(middle_layer, self.num_hidden_layers):
                    with tf.variable_scope("layer_%d" % layer_idx):
                        layer_input = prev_output
                        layer_output = self.build_layer(layer_input, attention_mask)
                    prev_output = layer_output

                    all_layer_outputs.append(layer_output)

            self.sequence_output = all_layer_outputs[-1]
            with tf.variable_scope("pooler"):
                first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
                self.pooled_output = tf.layers.dense(
                    first_token_tensor,
                    config.hidden_size,
                    activation=tf.tanh,
                    kernel_initializer=create_initializer(config.initializer_range))
            self.all_encoder_layers = all_layer_outputs

    def build_layer(self, layer_input, attention_mask):
        input_shape = get_shape_list(layer_input, expected_rank=3)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        input_width = input_shape[2]
        with tf.variable_scope("attention"):
            attention_heads = []
            with tf.variable_scope("self"):
                attention_head = attention_layer(
                    from_tensor=layer_input,
                    to_tensor=layer_input,
                    attention_mask=attention_mask,
                    num_attention_heads=self.num_attention_heads,
                    size_per_head=self.attention_head_size,
                    attention_probs_dropout_prob=self.attention_probs_dropout_prob,
                    initializer_range=self.initializer_range,
                    do_return_2d_tensor=False,
                    batch_size=batch_size,
                    from_seq_length=seq_length,
                    to_seq_length=seq_length)
                attention_heads.append(attention_head)

            attention_output = None
            if len(attention_heads) == 1:
                attention_output = attention_heads[0]
            else:
                # In the case where we have other sequences, we just concatenate
                # them to the self-attention head before the projection.
                attention_output = tf.concat(attention_heads, axis=-1)

            # Run a linear projection of `hidden_size` then add a residual
            # with `layer_input`.
            with tf.variable_scope("output"):
                attention_output = tf.layers.dense(
                    attention_output,
                    self.hidden_size,
                    kernel_initializer=create_initializer(self.initializer_range))
                attention_output = dropout(attention_output, self.hidden_dropout_prob)
                attention_output = layer_norm(attention_output + layer_input)
        # The activation is only applied to the "intermediate" hidden layer.
        with tf.variable_scope("intermediate"):
            intermediate_output = tf.layers.dense(
                attention_output,
                self.intermediate_size,
                activation=self.intermediate_act_fn,
                kernel_initializer=create_initializer(self.initializer_range))
        # Down-project back to `hidden_size` then add the residual.
        with tf.variable_scope("output"):
            layer_output = tf.layers.dense(
                intermediate_output,
                self.hidden_size,
                kernel_initializer=create_initializer(self.initializer_range))
            layer_output = dropout(layer_output, self.hidden_dropout_prob)
            layer_output = layer_norm(layer_output + attention_output)
        return layer_output

    def get_pooled_output(self):
        return self.pooled_output

    def get_sequence_output(self):
        """Gets final hidden layer of encoder.

        Returns:
          float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
          to the final hidden of the transformer encoder.
        """
        return self.sequence_output

    def get_all_encoder_layers(self):
        return self.all_encoder_layers

    def get_embedding_output(self):
        """Gets output of the embedding lookup (i.e., input to the transformer).

        Returns:
          float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
          to the output of the embedding layer, after summing the word
          embeddings with the positional embeddings and the token type embeddings,
          then performing layer normalization. This is the input to the transformer.
        """
        return self.embedding_output

    def get_embedding_table(self):
        return self.embedding_table

