from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.transformer.bert_common_v2 import *
from models.transformer.bert_common_v2 import self_attention, attention_layer
import copy
import tensorflow as tf


class DictReaderModel(object):
    def __init__(self,
                 config,
                 d_config,
                 is_training,
                 input_ids,
                 input_mask,
                 d_input_ids,
                 d_input_mask,
                 d_location_ids,
                 token_type_ids,
                 use_one_hot_embeddings=True,
                 use_target_pos_emb=False,
                 scope=None,
                 d_segment_ids=None ,
                 pool_dict_output=False):
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

        if d_segment_ids is None:
            d_segment_ids = d_input_mask

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

            with tf.compat.v1.variable_scope("dict"):
                with tf.compat.v1.variable_scope("embeddings"):
                    (self.d_embedding_output, self.d_embedding_table) = embedding_lookup(
                        input_ids=d_input_ids,
                        vocab_size=config.vocab_size,
                        embedding_size=config.hidden_size,
                        initializer_range=config.initializer_range,
                        word_embedding_name="word_embeddings",
                        use_one_hot_embeddings=use_one_hot_embeddings)

                    # Add positional embeddings and token type embeddings, then layer
                    # normalize and perform dropout.
                    if not use_target_pos_emb:
                        self.d_embedding_output = embedding_postprocessor(
                            input_tensor=self.d_embedding_output,
                            use_token_type=True,
                            token_type_ids=d_input_mask,
                            token_type_vocab_size=config.type_vocab_size,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=config.initializer_range,
                            max_position_embeddings=config.max_position_embeddings,
                            dropout_prob=config.hidden_dropout_prob)
                    else:
                        self.d_embedding_output = dict_embedding_processor(
                            input_tensor=self.d_embedding_output,
                            use_token_type=True,
                            token_type_ids=d_segment_ids,
                            token_type_vocab_size=d_config.type_vocab_size,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            target_loc_ids=d_location_ids,
                            max_target_position_embeddings=config.max_position_embeddings,
                            use_one_hot_embeddings=use_one_hot_embeddings,
                            initializer_range=d_config.initializer_range,
                            max_position_embeddings=d_config.max_position_embeddings,
                            dropout_prob=d_config.hidden_dropout_prob
                        )

            with tf.compat.v1.variable_scope("encoder"):
                self.all_encoder_layers, self.dict_layers = two_stack_transformer(
                    input_tensor_1=self.embedding_output,
                    input_mask_1=input_mask,
                    input_tensor_2=self.d_embedding_output,
                    input_mask_2=d_input_mask,
                    is_training=is_training,
                    config=config,
                    do_return_all_layers=True)

                self.sequence_output = self.all_encoder_layers[-1]
                with tf.compat.v1.variable_scope("pooler"):
                    first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
                    self.pooled_output = tf.keras.layers.Dense(config.hidden_size,
                                                               activation=tf.keras.activations.tanh,
                                                               kernel_initializer=create_initializer(config.initializer_range))(
                        first_token_tensor)

                self.dict_sequence_output = self.dict_layers[-1]
                if pool_dict_output:
                    with tf.compat.v1.variable_scope("dict"):
                        with tf.compat.v1.variable_scope("pooler"):
                            first_token_tensor = tf.squeeze(self.dict_sequence_output[:, 0:1, :], axis=1)
                            self.dict_pooled_output = tf.keras.layers.Dense(config.hidden_size,
                                                                       activation=tf.keras.activations.tanh,
                                                                       kernel_initializer=create_initializer(config.initializer_range))(
                                first_token_tensor)

    def get_pooled_output(self):
        return self.pooled_output

    def get_dict_pooled_output(self):
        return self.dict_pooled_output

    def get_sequence_output(self):
        return self.sequence_output

    def get_all_encoder_layers(self):
        return self.all_encoder_layers

    def get_embedding_output(self):
        return self.embedding_output

    def get_embedding_table(self):
        return self.embedding_table


# from : query is made
# to : key and value are made
def cross_attention(layer_input_from,
                    layer_input_to,
                   attention_mask,
                   config,
                   batch_size,
                   seq_length_from,
                   seq_length_to,
                   hidden_size,
                   initializer):
    attention_head_size = int(hidden_size / config.num_attention_heads)

    with tf.compat.v1.variable_scope("attention"):
        attention_heads = []
        with tf.compat.v1.variable_scope("self"):
            attention_head = attention_layer(
                from_tensor=layer_input_from,
                to_tensor=layer_input_to,
                attention_mask=attention_mask,
                num_attention_heads=config.num_attention_heads,
                size_per_head=attention_head_size,
                attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                initializer_range=config.initializer_range,
                do_return_2d_tensor=True,
                batch_size=batch_size,
                from_seq_length=seq_length_from,
                to_seq_length=seq_length_to)
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
            attention_output = dropout(attention_output, config.hidden_dropout_prob)
            attention_output = layer_norm(attention_output + layer_input_from)
    return attention_output


def dict_embedding_processor(input_tensor,
                             target_loc_ids,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            max_target_position_embeddings=512,
                            use_one_hot_embeddings=False,
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1):
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    width = input_shape[2]

    output = input_tensor

    if use_token_type:
        if token_type_ids is None:
            raise ValueError("`token_type_ids` must be specified if"
                                             "`use_token_type` is True.")
        token_type_table = tf.compat.v1.get_variable(
                name=token_type_embedding_name,
                shape=[token_type_vocab_size, width],
                initializer=create_initializer(initializer_range))
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
            full_position_embeddings = tf.compat.v1.get_variable(
                    name=position_embedding_name,
                    shape=[max_position_embeddings, width],
                    initializer=create_initializer(initializer_range))
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
            position_embeddings = tf.reshape(position_embeddings,
                                                                             position_broadcast_shape)
            output += position_embeddings

    target_position_embedding_table = tf.compat.v1.get_variable(
        name="target_position_embedding",
        shape=[max_target_position_embeddings, width],
        initializer=create_initializer(initializer_range))
    # Since the position embedding table is a learned variable, we create it
    # using a (long) sequence length `max_position_embeddings`. The actual
    # sequence length might be shorter than this, for faster training of
    # tasks that do not have long sequences.
    #
    # So `full_position_embeddings` is effectively an embedding table
    # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
    # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
    # perform a slice.
    if use_one_hot_embeddings:
        flat_input_ids = tf.reshape(target_loc_ids, [-1])
        one_hot_input_ids = tf.one_hot(flat_input_ids, depth=max_target_position_embeddings)
        target_pos_emb = tf.matmul(one_hot_input_ids, target_position_embedding_table)
        target_pos_emb = tf.reshape(target_pos_emb, [batch_size, -1, width])
    else:
        target_pos_emb = tf.nn.embedding_lookup(params=target_position_embedding_table, ids=target_loc_ids)

    # [batch, num_positions, hidden_dims]

    target_pos_emb = tf.reduce_sum(target_pos_emb, axis=1)

    # Only the last two dimensions are relevant (`seq_length` and `width`), so
    # we broadcast among the first dimensions, which is typically just
    # the batch size.
    target_pos_emb = tf.expand_dims(target_pos_emb, 1)
    output += target_pos_emb

    output = layer_norm_and_dropout(output, dropout_prob)
    return output




def two_stack_transformer(input_tensor_1,
                    input_mask_1,
                    input_tensor_2,
                    input_mask_2,
                    config,
                    is_training=True,
                    do_return_all_layers=False):
    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads

    if hidden_size % num_attention_heads != 0:
        raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))

    attention_head_size = int(hidden_size / num_attention_heads)
    input_shape_1 = get_shape_list(input_tensor_1, expected_rank=3)
    batch_size_1 = input_shape_1[0]
    seq_length_1 = input_shape_1[1]
    input_width_1 = input_shape_1[2]

    input_shape_2 = get_shape_list(input_tensor_2, expected_rank=3)
    batch_size_2 = input_shape_2[0]
    seq_length_2 = input_shape_2[1]
    input_width_2 = input_shape_2[2]

    initializer = create_initializer(config.initializer_range)

    attention_mask_1 = create_attention_mask_from_input_mask(
        input_tensor_1, input_mask_1)

    attention_mask_2 = create_attention_mask_from_input_mask(
        input_tensor_2, input_mask_2)

    attention_mask_2_to_1 = create_attention_mask_from_input_mask(
        input_tensor_2, input_mask_1)

    attention_mask_1_to_2 = create_attention_mask_from_input_mask(
        input_tensor_1, input_mask_2)

    # The Transformer performs sum residuals on all layers so the input needs
    # to be the same as the hidden size.
    if input_width_1 != hidden_size:
        raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                                         (input_width_1, hidden_size))

    # We keep the representation as a 2D tensor to avoid re-shaping it back and
    # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
    # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
    # help the optimizer.
    prev_output_1 = reshape_to_matrix(input_tensor_1)
    prev_output_2 = reshape_to_matrix(input_tensor_2)

    all_layer_outputs = []
    dict_layer_outputs = []
    for layer_idx in range(config.num_hidden_layers):
        with tf.compat.v1.variable_scope("layer_%d" % layer_idx):
            layer_input_1 = prev_output_1
            layer_input_2 = prev_output_2

            attention_output_1 = self_attention(layer_input_1,
                                                attention_mask_1,
                                                config,
                                                batch_size_1,
                                                seq_length_1,
                                                hidden_size,
                                                initializer)
            with tf.compat.v1.variable_scope("dict"):
                attention_output_2 = self_attention(layer_input_2,
                                                    attention_mask_2,
                                                    config,
                                                    batch_size_2,
                                                    seq_length_2,
                                                    hidden_size,
                                                    initializer)

            with tf.compat.v1.variable_scope("cross_1_to_2"):
                attention_output_1 = cross_attention(attention_output_1,
                                                     layer_input_2,
                                                     attention_mask_1_to_2,
                                                     config,
                                                     batch_size_2,
                                                     seq_length_1,
                                                     seq_length_2,
                                                     hidden_size,
                                                     initializer)

            with tf.compat.v1.variable_scope("cross_2_to_1"):
                attention_output_2 = cross_attention(attention_output_2,
                                                     layer_input_1,
                                                     attention_mask_2_to_1,
                                                     config,
                                                     batch_size_2,
                                                     seq_length_2,
                                                     seq_length_1,
                                                     hidden_size,
                                                     initializer)

            # The activation is only applied to the "intermediate" hidden layer.
            with tf.compat.v1.variable_scope("intermediate"):
                intermediate_output_1 = dense(config.intermediate_size, initializer,
                                            activation=get_activation(config.hidden_act))(attention_output_1)

            # Down-project back to `hidden_size` then add the residual.
            with tf.compat.v1.variable_scope("output"):
                layer_output_1 = dense(hidden_size, initializer)(intermediate_output_1)
                layer_output_1 = dropout(layer_output_1, config.hidden_dropout_prob)
                layer_output_1 = layer_norm(layer_output_1 + attention_output_1)
                prev_output_1 = layer_output_1
                all_layer_outputs.append(layer_output_1)


            with tf.compat.v1.variable_scope("dict"):
                with tf.compat.v1.variable_scope("intermediate"):
                    intermediate_output_2 = dense(config.intermediate_size, initializer,
                                                activation=get_activation(config.hidden_act))(attention_output_2)

                # Down-project back to `hidden_size` then add the residual.
                with tf.compat.v1.variable_scope("output"):
                    layer_output_2 = dense(hidden_size, initializer)(intermediate_output_2)
                    layer_output_2 = dropout(layer_output_2, config.hidden_dropout_prob)
                    layer_output_2 = layer_norm(layer_output_2 + attention_output_2)
                    prev_output_2 = layer_output_2
                    dict_layer_outputs.append(layer_output_2)

    if do_return_all_layers:
        final_outputs = []
        for layer_output in all_layer_outputs:
            final_output = reshape_from_matrix(layer_output, input_shape_1)
            final_outputs.append(final_output)

        dict_layers = []
        for layer in dict_layer_outputs:
            l = reshape_from_matrix(layer, input_shape_2)
            dict_layers.append(l)

        return final_outputs, dict_layers
    else:
        final_output = reshape_from_matrix(prev_output_1, input_shape_1)
        dict_layer = reshape_from_matrix(prev_output_2, input_shape_1)
        return final_output, dict_layer



def limted_interaction_transformer(input_tensor_1,
                    input_mask_1,
                    input_tensor_2,
                    input_mask_2,
                    config,
                    is_training=True,
                    do_return_all_layers=False):
    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads

    if hidden_size % num_attention_heads != 0:
        raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))

    attention_head_size = int(hidden_size / num_attention_heads)
    input_shape_1 = get_shape_list(input_tensor_1, expected_rank=3)
    batch_size_1 = input_shape_1[0]
    seq_length_1 = input_shape_1[1]
    input_width_1 = input_shape_1[2]

    input_shape_2 = get_shape_list(input_tensor_2, expected_rank=3)
    batch_size_2 = input_shape_2[0]
    seq_length_2 = input_shape_2[1]
    input_width_2 = input_shape_2[2]

    initializer = create_initializer(config.initializer_range)

    attention_mask_1 = create_attention_mask_from_input_mask(
        input_tensor_1, input_mask_1)

    attention_mask_2 = create_attention_mask_from_input_mask(
        input_tensor_2, input_mask_2)

    attention_mask_2_to_1 = create_attention_mask_from_input_mask(
        input_tensor_2, input_mask_1)

    attention_mask_1_to_2 = create_attention_mask_from_input_mask(
        input_tensor_1, input_mask_2)

    # The Transformer performs sum residuals on all layers so the input needs
    # to be the same as the hidden size.
    if input_width_1 != hidden_size:
        raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                                         (input_width_1, hidden_size))

    # We keep the representation as a 2D tensor to avoid re-shaping it back and
    # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
    # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
    # help the optimizer.
    prev_output_1 = reshape_to_matrix(input_tensor_1)
    prev_output_2 = reshape_to_matrix(input_tensor_2)

    all_layer_outputs = []
    dict_layer_outputs = []
    for layer_idx in range(config.num_hidden_layers):
        with tf.compat.v1.variable_scope("layer_%d" % layer_idx):
            layer_input_1 = prev_output_1
            layer_input_2 = prev_output_2

            attention_output_1 = self_attention(layer_input_1,
                                                attention_mask_1,
                                                config,
                                                batch_size_1,
                                                seq_length_1,
                                                hidden_size,
                                                initializer)
            with tf.compat.v1.variable_scope("dict"):
                attention_output_2 = self_attention(layer_input_2,
                                                    attention_mask_2,
                                                    config,
                                                    batch_size_2,
                                                    seq_length_2,
                                                    hidden_size,
                                                    initializer)

            with tf.compat.v1.variable_scope("cross_1_to_2"):
                attention_output_1 = cross_attention(attention_output_1,
                                                     layer_input_2,
                                                     attention_mask_1_to_2,
                                                     config,
                                                     batch_size_2,
                                                     seq_length_1,
                                                     seq_length_2,
                                                     hidden_size,
                                                     initializer)

            with tf.compat.v1.variable_scope("cross_2_to_1"):
                attention_output_2 = cross_attention(attention_output_2,
                                                     layer_input_1,
                                                     attention_mask_2_to_1,
                                                     config,
                                                     batch_size_2,
                                                     seq_length_2,
                                                     seq_length_1,
                                                     hidden_size,
                                                     initializer)

            # The activation is only applied to the "intermediate" hidden layer.
            with tf.compat.v1.variable_scope("intermediate"):
                intermediate_output_1 = dense(config.intermediate_size, initializer,
                                            activation=get_activation(config.hidden_act))(attention_output_1)

            # Down-project back to `hidden_size` then add the residual.
            with tf.compat.v1.variable_scope("output"):
                layer_output_1 = dense(hidden_size, initializer)(intermediate_output_1)
                layer_output_1 = dropout(layer_output_1, config.hidden_dropout_prob)
                layer_output_1 = layer_norm(layer_output_1 + attention_output_1)
                prev_output_1 = layer_output_1
                all_layer_outputs.append(layer_output_1)


            with tf.compat.v1.variable_scope("dict"):
                with tf.compat.v1.variable_scope("intermediate"):
                    intermediate_output_2 = dense(config.intermediate_size, initializer,
                                                activation=get_activation(config.hidden_act))(attention_output_2)

                # Down-project back to `hidden_size` then add the residual.
                with tf.compat.v1.variable_scope("output"):
                    layer_output_2 = dense(hidden_size, initializer)(intermediate_output_2)
                    layer_output_2 = dropout(layer_output_2, config.hidden_dropout_prob)
                    layer_output_2 = layer_norm(layer_output_2 + attention_output_2)
                    prev_output_2 = layer_output_2
                    dict_layer_outputs.append(layer_output_2)

    if do_return_all_layers:
        final_outputs = []
        for layer_output in all_layer_outputs:
            final_output = reshape_from_matrix(layer_output, input_shape_1)
            final_outputs.append(final_output)

        dict_layers = []
        for layer in dict_layer_outputs:
            l = reshape_from_matrix(layer, input_shape_2)
            dict_layers.append(l)

        return final_outputs, dict_layers
    else:
        final_output = reshape_from_matrix(prev_output_1, input_shape_1)
        dict_layer = reshape_from_matrix(prev_output_2, input_shape_1)
        return final_output, dict_layer
