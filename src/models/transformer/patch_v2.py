# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The main BERT model and related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import re
import six
import tensorflow as tf

from models.transformer.bert_common_v2 import gelu, get_activation, dropout, layer_norm, create_initializer, \
    embedding_lookup, embedding_postprocessor, create_attention_mask_from_input_mask, get_shape_list, reshape_to_matrix, \
    reshape_from_matrix


class BertConfig(object):
    """Configuration for `BertModel`."""

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 mr_layer=1,
                 mr_num_route=10,
                 mr_key_layer=0,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler.
            hidden_dropout_prob: The dropout probability for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The stdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.mr_layer = mr_layer
        self.mr_num_route = mr_num_route
        self.mr_key_layer = mr_key_layer
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with tf.io.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BertModel(object):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Example usage:

    ```python
    # Already been converted into WordPiece token ids
    input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
    input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
    token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = modeling.BertModel(config=config, is_training=True,
        input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

    label_embeddings = tf.get_variable(...)
    pooled_output = model.get_pooled_output()
    logits = tf.matmul(pooled_output, label_embeddings)
    ...
    ```
    """

    def __init__(self,
                             config,
                             is_training,
                             input_ids,
                             input_mask=None,
                             token_type_ids=None,
                             use_one_hot_embeddings=True,
                             scope=None):
        """Constructor for BertModel.

        Args:
            config: `BertConfig` instance.
            is_training: bool. rue for training model, false for eval model. Controls
                whether dropout will be applied.
            input_ids: int32 Tensor of shape [batch_size, seq_length].
            input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
            token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
            use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
                embeddings or tf.embedding_lookup() for the word embeddings. On the TPU,
                it is must faster if this is True, on the CPU or GPU, it is faster if
                this is False.
            scope: (optional) variable scope. Defaults to "bert".

        Raises:
            ValueError: The config is invalid or one of the input tensor shapes
                is invalid.
        """
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

        with tf.compat.v1.variable_scope(scope, default_name="bert"):
            with tf.compat.v1.variable_scope("embeddings"):
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

            with tf.compat.v1.variable_scope("encoder"):
                # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
                # mask of shape [batch_size, seq_length, seq_length] which is used
                # for the attention scores.
                attention_mask = create_attention_mask_from_input_mask(
                        input_ids, input_mask)

                # Run the stacked transformer.
                # `sequence_output` shape = [batch_size, seq_length, hidden_size].
                self.all_encoder_layers, key = transformer_model(
                        input_tensor=self.embedding_output,
                        attention_mask=attention_mask,
                        input_mask=input_mask,
                        hidden_size=config.hidden_size,
                        num_hidden_layers=config.num_hidden_layers,
                        num_attention_heads=config.num_attention_heads,
                        is_training=is_training,
                        #mr_layer=config.mr_layer,
                        mr_num_route=config.mr_num_route,
                        #mr_key_layer=config.mr_key_layer,
                        intermediate_size=config.intermediate_size,
                        intermediate_act_fn=get_activation(config.hidden_act),
                        hidden_dropout_prob=config.hidden_dropout_prob,
                        attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                        initializer_range=config.initializer_range,
                        do_return_all_layers=True)

            self.key = key
            self.sequence_output = self.all_encoder_layers[-1]
            # The "pooler" converts the encoded sequence tensor of shape
            # [batch_size, seq_length, hidden_size] to a tensor of shape
            # [batch_size, hidden_size]. This is necessary for segment-level
            # (or segment-pair-level) classification tasks where we need a fixed
            # dimensional representation of the segment.
            with tf.compat.v1.variable_scope("pooler"):
                # We "pool" the model by simply taking the hidden state corresponding
                # to the first token. We assume that this has been pre-trained
                first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
                self.pooled_output = tf.keras.layers.Dense(config.hidden_size,
                                      activation=tf.keras.activations.tanh,
                                      kernel_initializer=create_initializer(config.initializer_range))(first_token_tensor)
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


def attention_layer(from_tensor,
                                        to_tensor,
                                        attention_mask=None,
                                        num_attention_heads=1,
                                        size_per_head=512,
                                        query_act=None,
                                        key_act=None,
                                        value_act=None,
                                        attention_probs_dropout_prob=0.0,
                                        initializer_range=0.02,
                                        do_return_2d_tensor=False,
                                        batch_size=None,
                                        from_seq_length=None,
                                        to_seq_length=None):
    """Performs multi-headed attention from `from_tensor` to `to_tensor`.

    This is an implementation of multi-headed attention based on "Attention
    is all you Need". If `from_tensor` and `to_tensor` are the same, then
    this is self-attention. Each timestep in `from_tensor` attends to the
    corresponding sequence in `to_tensor`, and returns a fixed-with vector.

    This function first projects `from_tensor` into a "query" tensor and
    `to_tensor` into "key" and "value" tensors. These are (effectively) a list
    of tensors of length `num_attention_heads`, where each tensor is of shape
    [batch_size, seq_length, size_per_head].

    Then, the query and key tensors are dot-producted and scaled. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities, then concatenated back to a single
    tensor and returned.

    In practice, the multi-headed attention are done with transposes and
    reshapes rather than actual separate tensors.

    Args:
        from_tensor: float Tensor of shape [batch_size, from_seq_length,
            from_width].
        to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
        attention_mask: (optional) int32 Tensor of shape [batch_size,
            from_seq_length, to_seq_length]. The values should be 1 or 0. The
            attention scores will effectively be set to -infinity for any positions in
            the mask that are 0, and will be unchanged for positions that are 1.
        num_attention_heads: int. Number of attention heads.
        size_per_head: int. Size of each attention head.
        query_act: (optional) Activation function for the query transform.
        key_act: (optional) Activation function for the key transform.
        value_act: (optional) Activation function for the value transform.
        attention_probs_dropout_prob: (optional) float. Dropout probability of the
            attention probabilities.
        initializer_range: float. Range of the weight initializer.
        do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
            * from_seq_length, num_attention_heads * size_per_head]. If False, the
            output will be of shape [batch_size, from_seq_length, num_attention_heads
            * size_per_head].
        batch_size: (Optional) int. If the input is 2D, this might be the batch size
            of the 3D version of the `from_tensor` and `to_tensor`.
        from_seq_length: (Optional) If the input is 2D, this might be the seq length
            of the 3D version of the `from_tensor`.
        to_seq_length: (Optional) If the input is 2D, this might be the seq length
            of the 3D version of the `to_tensor`.

    Returns:
        float Tensor of shape [batch_size, from_seq_length,
            num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
            true, this will be of shape [batch_size * from_seq_length,
            num_attention_heads * size_per_head]).

    Raises:
        ValueError: Any of the arguments or tensor shapes are invalid.
    """

    def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                                                     seq_length, width):
        output_tensor = tf.reshape(
                input_tensor, [batch_size, seq_length, num_attention_heads, width])

        output_tensor = tf.transpose(a=output_tensor, perm=[0, 2, 1, 3])
        return output_tensor

    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

    if len(from_shape) != len(to_shape):
        raise ValueError(
                "The rank of `from_tensor` must match the rank of `to_tensor`.")

    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]
    elif len(from_shape) == 2:
        if (batch_size is None or from_seq_length is None or to_seq_length is None):
            raise ValueError(
                    "When passing in rank 2 tensors to attention_layer, the values "
                    "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                    "must all be specified.")

    # Scalar dimensions referenced here:
    #     B = batch size (number of sequences)
    #     F = `from_tensor` sequence length
    #     T = `to_tensor` sequence length
    #     N = `num_attention_heads`
    #     H = `size_per_head`

    from_tensor_2d = reshape_to_matrix(from_tensor)
    to_tensor_2d = reshape_to_matrix(to_tensor)

    # `query_layer` = [B*F, N*H]
    query_layer = tf.keras.layers.Dense(
            num_attention_heads * size_per_head,
            activation=query_act,
            name="query",
            kernel_initializer=create_initializer(initializer_range))(from_tensor_2d)

    # `key_layer` = [B*T, N*H]
    key_layer = tf.keras.layers.Dense(
            num_attention_heads * size_per_head,
            activation=key_act,
            name="key",
            kernel_initializer=create_initializer(initializer_range))(to_tensor_2d)

    # `value_layer` = [B*T, N*H]
    value_layer = tf.keras.layers.Dense(
            num_attention_heads * size_per_head,
            activation=value_act,
            name="value",
            kernel_initializer=create_initializer(initializer_range))(to_tensor_2d)

    # `query_layer` = [B, N, F, H]
    query_layer = transpose_for_scores(query_layer, batch_size,
                                                                         num_attention_heads, from_seq_length,
                                                                         size_per_head)

    # `key_layer` = [B, N, T, H]
    key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                                                     to_seq_length, size_per_head)

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    # `attention_scores` = [B, N, F, T]
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores,
                                                                 1.0 / math.sqrt(float(size_per_head)))

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
    attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

    # `value_layer` = [B, T, N, H]
    value_layer = tf.reshape(
            value_layer,
            [batch_size, to_seq_length, num_attention_heads, size_per_head])

    # `value_layer` = [B, N, T, H]
    value_layer = tf.transpose(a=value_layer, perm=[0, 2, 1, 3])

    # `context_layer` = [B, N, F, H]
    context_layer = tf.matmul(attention_probs, value_layer)

    # `context_layer` = [B, F, N, H]
    context_layer = tf.transpose(a=context_layer, perm=[0, 2, 1, 3])

    if do_return_2d_tensor:
        # `context_layer` = [B*F, N*V]
        context_layer = tf.reshape(
                context_layer,
                [batch_size * from_seq_length, num_attention_heads * size_per_head])
    else:
        # `context_layer` = [B, F, N*V]
        context_layer = tf.reshape(
                context_layer,
                [batch_size, from_seq_length, num_attention_heads * size_per_head])

    return context_layer




EXT_QUERY_IN = 0
EXT_QUERY_OUT = 1
EXT_KEY_IN = 2
EXT_KEY_OUT = 3
EXT_VALUE_IN = 4
EXT_VALUE_OUT = 5
EXT_ATT_OUT = 6
EXT_ATT_PROJ = 7
EXT_LAYER_OUT = 8
####
EXT_SIZE = 9

def attention_layer_w_ext(from_tensor,
                                        to_tensor,
                                        attention_mask=None,
                                        num_attention_heads=1,
                                        size_per_head=512,
                                        ext_slice=None, # [Num_tokens, n_items, hidden_dim]
                                        query_act=None,
                                        key_act=None,
                                        value_act=None,
                                        attention_probs_dropout_prob=0.0,
                                        initializer_range=0.02,
                                        do_return_2d_tensor=False,
                                        batch_size=None,
                                        from_seq_length=None,
                                        to_seq_length=None):
    """Performs multi-headed attention from `from_tensor` to `to_tensor`.

    This is an implementation of multi-headed attention based on "Attention
    is all you Need". If `from_tensor` and `to_tensor` are the same, then
    this is self-attention. Each timestep in `from_tensor` attends to the
    corresponding sequence in `to_tensor`, and returns a fixed-with vector.

    This function first projects `from_tensor` into a "query" tensor and
    `to_tensor` into "key" and "value" tensors. These are (effectively) a list
    of tensors of length `num_attention_heads`, where each tensor is of shape
    [batch_size, seq_length, size_per_head].

    Then, the query and key tensors are dot-producted and scaled. These are
    softmaxed to obtain attention probabilities. The value tensors are then
    interpolated by these probabilities, then concatenated back to a single
    tensor and returned.

    In practice, the multi-headed attention are done with transposes and
    reshapes rather than actual separate tensors.

    Args:
        from_tensor: float Tensor of shape [batch_size, from_seq_length,
            from_width].
        to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
        attention_mask: (optional) int32 Tensor of shape [batch_size,
            from_seq_length, to_seq_length]. The values should be 1 or 0. The
            attention scores will effectively be set to -infinity for any positions in
            the mask that are 0, and will be unchanged for positions that are 1.
        num_attention_heads: int. Number of attention heads.
        size_per_head: int. Size of each attention head.
        query_act: (optional) Activation function for the query transform.
        key_act: (optional) Activation function for the key transform.
        value_act: (optional) Activation function for the value transform.
        attention_probs_dropout_prob: (optional) float. Dropout probability of the
            attention probabilities.
        initializer_range: float. Range of the weight initializer.
        do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
            * from_seq_length, num_attention_heads * size_per_head]. If False, the
            output will be of shape [batch_size, from_seq_length, num_attention_heads
            * size_per_head].
        batch_size: (Optional) int. If the input is 2D, this might be the batch size
            of the 3D version of the `from_tensor` and `to_tensor`.
        from_seq_length: (Optional) If the input is 2D, this might be the seq length
            of the 3D version of the `from_tensor`.
        to_seq_length: (Optional) If the input is 2D, this might be the seq length
            of the 3D version of the `to_tensor`.

    Returns:
        float Tensor of shape [batch_size, from_seq_length,
            num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
            true, this will be of shape [batch_size * from_seq_length,
            num_attention_heads * size_per_head]).

    Raises:
        ValueError: Any of the arguments or tensor shapes are invalid.
    """

    def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                                                     seq_length, width):
        output_tensor = tf.reshape(
                input_tensor, [batch_size, seq_length, num_attention_heads, width])

        output_tensor = tf.transpose(a=output_tensor, perm=[0, 2, 1, 3])
        return output_tensor

    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

    if len(from_shape) != len(to_shape):
        raise ValueError(
                "The rank of `from_tensor` must match the rank of `to_tensor`.")

    if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]
    elif len(from_shape) == 2:
        if (batch_size is None or from_seq_length is None or to_seq_length is None):
            raise ValueError(
                    "When passing in rank 2 tensors to attention_layer, the values "
                    "for `batch_size`, `from_seq_length`, and `to_seq_length` "
                    "must all be specified.")

    # Scalar dimensions referenced here:
    #     B = batch size (number of sequences)
    #     F = `from_tensor` sequence length
    #     T = `to_tensor` sequence length
    #     N = `num_attention_heads`
    #     H = `size_per_head`

    from_tensor_2d = reshape_to_matrix(from_tensor)
    to_tensor_2d = reshape_to_matrix(to_tensor)

    def get_ext_slice(idx):
        return ext_slice[:, idx, :]

    print("from_tensor_2d ", from_tensor_2d.shape)

    query_in = from_tensor_2d + get_ext_slice(EXT_QUERY_IN)
    query_in = from_tensor_2d

    # `query_layer` = [B*F, N*H]
    query_layer = tf.keras.layers.Dense(
            num_attention_heads * size_per_head,
            activation=query_act,
            name="query",
            kernel_initializer=create_initializer(initializer_range))(query_in)

    query_layer = query_layer + get_ext_slice(EXT_QUERY_OUT)

    key_in = to_tensor_2d
    key_in = to_tensor_2d + get_ext_slice(EXT_KEY_IN)
    # `key_layer` = [B*T, N*H]
    key_layer = tf.keras.layers.Dense(
            num_attention_heads * size_per_head,
            activation=key_act,
            name="key",
            kernel_initializer=create_initializer(initializer_range))(key_in)

    key_layer = key_layer + get_ext_slice(EXT_KEY_OUT)

    value_in = to_tensor_2d
    value_in = to_tensor_2d + get_ext_slice(EXT_VALUE_IN)
    # `value_layer` = [B*T, N*H]
    value_layer = tf.keras.layers.Dense(
            num_attention_heads * size_per_head,
            activation=value_act,
            name="value",
            kernel_initializer=create_initializer(initializer_range))(value_in)

    value_layer = value_layer + get_ext_slice(EXT_VALUE_OUT)

    # `query_layer` = [B, N, F, H]
    query_layer = transpose_for_scores(query_layer, batch_size,
                                                     num_attention_heads, from_seq_length,
                                                     size_per_head)

    # `key_layer` = [B, N, T, H]
    key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                                                     to_seq_length, size_per_head)

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    # `attention_scores` = [B, N, F, T]
    attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
    attention_scores = tf.multiply(attention_scores,
                                                                 1.0 / math.sqrt(float(size_per_head)))

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
    attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

    # `value_layer` = [B, T, N, H]
    value_layer = tf.reshape(
            value_layer,
            [batch_size, to_seq_length, num_attention_heads, size_per_head])

    # `value_layer` = [B, N, T, H]
    value_layer = tf.transpose(a=value_layer, perm=[0, 2, 1, 3])

    # `context_layer` = [B, N, F, H]
    context_layer = tf.matmul(attention_probs, value_layer)

    # `context_layer` = [B, F, N, H]
    context_layer = tf.transpose(a=context_layer, perm=[0, 2, 1, 3])

    if do_return_2d_tensor:
        # `context_layer` = [B*F, N*V]
        context_layer = tf.reshape(
                context_layer,
                [batch_size * from_seq_length, num_attention_heads * size_per_head])
    else:
        # `context_layer` = [B, F, N*V]
        context_layer = tf.reshape(
                context_layer,
                [batch_size, from_seq_length, num_attention_heads * size_per_head])

    return context_layer


def dense(hidden_units, initializer, activation=None):
    if activation is not None:
        return tf.keras.layers.Dense(hidden_units,
                                     kernel_initializer=initializer,
                                     activation=activation)
    else:
        return tf.keras.layers.Dense(hidden_units,
                                     kernel_initializer=initializer,
                                     )



def transformer_model(input_tensor,
                    attention_mask=None,
                    input_mask=None,
                    hidden_size=768,
                    num_hidden_layers=12,
                    num_attention_heads=12,
                    mr_num_route=10,
                    intermediate_size=3072,
                    intermediate_act_fn=gelu,
                    hidden_dropout_prob=0.1,
                    attention_probs_dropout_prob=0.1,
                    initializer_range=0.02,
                    is_training=True,
                    do_return_all_layers=False):
    """Multi-headed, multi-layer Transformer from "Attention is All You Need".

    This is almost an exact implementation of the original Transformer encoder.

    See the original paper:
    https://arxiv.org/abs/1706.03762

    Also see:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

    Args:
        input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
        attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
            seq_length], with 1 for positions that can be attended to and 0 in
            positions that should not be.
        hidden_size: int. Hidden size of the Transformer.
        num_hidden_layers: int. Number of layers (blocks) in the Transformer.
        num_attention_heads: int. Number of attention heads in the Transformer.
        intermediate_size: int. The size of the "intermediate" (a.k.a., feed
            forward) layer.
        intermediate_act_fn: function. The non-linear activation function to apply
            to the output of the intermediate/feed-forward layer.
        hidden_dropout_prob: float. Dropout probability for the hidden layers.
        attention_probs_dropout_prob: float. Dropout probability of the attention
            probabilities.
        initializer_range: float. Range of the initializer (stddev of truncated
            normal).
        do_return_all_layers: Whether to also return all layers or just the final
            layer.

    Returns:
        float Tensor of shape [batch_size, seq_length, hidden_size], the final
        hidden layer of the Transformer.

    Raises:
        ValueError: A Tensor shape or parameter is invalid.
    """
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))

    attention_head_size = int(hidden_size / num_attention_heads)
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    input_width = input_shape[2]

    initializer = create_initializer(initializer_range)

    ext_tensor = tf.compat.v1.get_variable("ext_tensor",
                                 shape=[num_hidden_layers, mr_num_route, EXT_SIZE ,hidden_size],
                                 initializer=initializer,
                                 )
    ext_tensor_inter = tf.compat.v1.get_variable("ext_tensor_inter",
                                       shape=[num_hidden_layers, mr_num_route, intermediate_size],
                                       initializer=initializer,
                                           )
    # The Transformer performs sum residuals on all layers so the input needs
    # to be the same as the hidden size.
    if input_width != hidden_size:
        raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                                         (input_width, hidden_size))

    # We keep the representation as a 2D tensor to avoid re-shaping it back and
    # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
    # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
    # help the optimizer.
    prev_output = reshape_to_matrix(input_tensor)

    def is_mr_layer(layer_idx):
        if layer_idx > 1:
            return True
        else:
            return False

    all_layer_outputs = []
    for layer_idx in range(num_hidden_layers):
        if not is_mr_layer(layer_idx):
            with tf.compat.v1.variable_scope("layer_%d" % layer_idx):
                layer_input = prev_output

                with tf.compat.v1.variable_scope("attention"):
                    attention_heads = []
                    with tf.compat.v1.variable_scope("self"):
                        attention_head = attention_layer(
                                from_tensor=layer_input,
                                to_tensor=layer_input,
                                attention_mask=attention_mask,
                                num_attention_heads=num_attention_heads,
                                size_per_head=attention_head_size,
                                attention_probs_dropout_prob=attention_probs_dropout_prob,
                                initializer_range=initializer_range,
                                do_return_2d_tensor=True,
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
                    with tf.compat.v1.variable_scope("output"):
                        attention_output = dense(hidden_size, initializer)(attention_output)
                        attention_output = dropout(attention_output, hidden_dropout_prob)
                        attention_output = layer_norm(attention_output + layer_input)

                # The activation is only applied to the "intermediate" hidden layer.
                with tf.compat.v1.variable_scope("intermediate"):
                    intermediate_output = dense(intermediate_size, initializer,
                                                activation=intermediate_act_fn)(attention_output)

                # Down-project back to `hidden_size` then add the residual.
                with tf.compat.v1.variable_scope("output"):
                    layer_output = dense(hidden_size, initializer)(intermediate_output)
                    layer_output = dropout(layer_output, hidden_dropout_prob)
                    layer_output = layer_norm(layer_output + attention_output)
                    prev_output = layer_output
                    all_layer_outputs.append(layer_output)

                with tf.compat.v1.variable_scope("mr_key"):
                    key_output = tf.keras.layers.Dense(
                        mr_num_route,
                        kernel_initializer=create_initializer(initializer_range))(intermediate_output)
                    key_output = dropout(key_output, hidden_dropout_prob)

                    if is_training:
                        key = tf.random.categorical(key_output, 1) # [batch_size, 1]
                        key = tf.reshape(key, [-1])
                    else:
                        key = tf.math.argmax(input=key_output, axis=1)

        else: # Case MR layer
            with tf.compat.v1.variable_scope("layer_%d" % layer_idx):
                layer_input = prev_output
                ext_slice = tf.gather(ext_tensor[layer_idx], key)
                ext_interm_slice = tf.gather(ext_tensor_inter[layer_idx], key)
                print("ext_slice (batch*seq, ", ext_slice.shape)
                with tf.compat.v1.variable_scope("attention"):
                    attention_heads = []
                    with tf.compat.v1.variable_scope("self"):
                        attention_head = attention_layer_w_ext(
                            from_tensor=layer_input,
                            to_tensor=layer_input,
                            attention_mask=attention_mask,
                            ext_slice=ext_slice,
                            num_attention_heads=num_attention_heads,
                            size_per_head=attention_head_size,
                            attention_probs_dropout_prob=attention_probs_dropout_prob,
                            initializer_range=initializer_range,
                            do_return_2d_tensor=True,
                            batch_size=batch_size,
                            from_seq_length=seq_length,
                            to_seq_length=seq_length)
                        attention_head = attention_head + ext_slice[:,EXT_ATT_OUT,:]
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
                    with tf.compat.v1.variable_scope("output"):
                        attention_output = dense(hidden_size, initializer)(attention_output)
                        attention_output = dropout(attention_output, hidden_dropout_prob)
                        attention_output = attention_output + ext_slice[:,EXT_ATT_PROJ,:]
                        attention_output = layer_norm(attention_output + layer_input)

                # The activation is only applied to the "intermediate" hidden layer.
                with tf.compat.v1.variable_scope("intermediate"):
                    intermediate_output = dense(intermediate_size, initializer,
                                                activation=intermediate_act_fn)(attention_output)
                    intermediate_output = ext_interm_slice + intermediate_output
                # Down-project back to `hidden_size` then add the residual.
                with tf.compat.v1.variable_scope("output"):
                    layer_output = dense(hidden_size, initializer)(intermediate_output)
                    layer_output = layer_output + ext_slice[:, EXT_LAYER_OUT,:]
                    layer_output = dropout(layer_output, hidden_dropout_prob)
                    layer_output = layer_norm(layer_output + attention_output)
                    prev_output = layer_output
                    all_layer_outputs.append(layer_output)

    if do_return_all_layers:
        final_outputs = []
        for layer_output in all_layer_outputs:
            final_output = reshape_from_matrix(layer_output, input_shape)
            final_outputs.append(final_output)
        return final_outputs, key
    else:
        final_output = reshape_from_matrix(prev_output, input_shape)
        return final_output, key

