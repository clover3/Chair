import copy

import tensorflow as tf

import models.transformer.bert_common_v2 as bc
import tlm.model.base as base
from misc.categorical_gradient import gather, categorical_sampling


class SSDR(base.BertModelInterface):
    def __init__(self,
                 config,
                 ssdr_config,
                 is_training,
                 input_ids,
                 input_mask,
                 token_type_ids,
                 d_input_ids,
                 d_input_mask,
                 d_segment_ids,
                 d_location_ids,
                 ab_mapping,
                 use_one_hot_embeddings=True,
                 scope=None,
                 ):
        super(SSDR, self).__init__()

        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        batch_size, seq_length = get_batch_and_seq_length(input_ids, 2)

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

        if d_segment_ids is None:
            d_segment_ids = d_input_mask

        with tf.compat.v1.variable_scope(scope, default_name="bert"):
            self.main_transformer = MainTransformer(
                config, ssdr_config, input_ids, input_mask, token_type_ids, use_one_hot_embeddings)

            key_out = self.main_transformer.build_key()
            self.dict_tranformer = SecondTransformer(
                config, ssdr_config, d_input_ids, d_input_mask, d_segment_ids, use_one_hot_embeddings)

            aligned_key = tf.gather(key_out, ab_mapping)

            scores, info_vectors = self.dict_tranformer.build(aligned_key)
            info_vector = select_value(batch_size, ab_mapping, scores, info_vectors, "sample")

            all_encoder_layers = self.main_transformer.build_remain(info_vector, d_location_ids)

            self.all_encoder_layers = all_encoder_layers
            self.sequence_output = self.all_encoder_layers[-1]
            self.pooled_output = get_pooler(self.sequence_output, config)
            self.embedding_output = self.main_transformer.embedding_output
            self.embedding_table = self.main_transformer.embedding_table


def get_batch_and_seq_length(input_ids, expected_rank):
    input_shape = bc.get_shape_list(input_ids, expected_rank=expected_rank)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    return batch_size, seq_length


def get_pooler(sequence_output, config):
    with tf.compat.v1.variable_scope("pooler"):
        first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
        pooled_output = tf.keras.layers.Dense(config.hidden_size,
                                                   activation=tf.keras.activations.tanh,
                                                   kernel_initializer=bc.create_initializer(config.initializer_range))(
            first_token_tensor)
    return pooled_output



class TransformerBase:
    def __init__(self, config, input_ids, input_mask, segment_ids, use_one_hot_embeddings):
        self.config = config
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.batch_size, self.seq_length = get_batch_and_seq_length(input_ids, 2)
        self.initializer = base.create_initializer(config.initializer_range)
        self.attention_mask = bc.create_attention_mask_from_input_mask(
            input_ids, self.input_mask)

    def forward_layer(self, prev_output):
        hidden_size = self.config.hidden_size
        layer_input = prev_output
        attention_output = bc.self_attention(layer_input,
                                             self.attention_mask,
                                             self.config,
                                             self.batch_size,
                                             self.seq_length,
                                             hidden_size,
                                             self.initializer)

        with tf.compat.v1.variable_scope("intermediate"):
            intermediate_output = bc.dense(self.config.intermediate_size, self.initializer,
                                           activation=bc.get_activation(self.config.hidden_act))(attention_output)

        with tf.compat.v1.variable_scope("output"):
            layer_output = bc.dense(hidden_size, self.initializer)(intermediate_output)
            layer_output = bc.dropout(layer_output, self.config.hidden_dropout_prob)
            layer_output = bc.layer_norm(layer_output + attention_output)
            prev_output = layer_output
        return intermediate_output, layer_output

    def get_embeddings(self, input_ids, segment_ids):
        config = self.config
        # Perform embedding lookup on the word ids.
        (self.embedding_output, self.embedding_table) = bc.embedding_lookup(
            input_ids=input_ids,
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            initializer_range=config.initializer_range,
            word_embedding_name="word_embeddings",
            use_one_hot_embeddings=self.use_one_hot_embeddings)

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        self.embedding_output = bc.embedding_postprocessor(
            input_tensor=self.embedding_output,
            use_token_type=True,
            token_type_ids=segment_ids,
            token_type_vocab_size=config.type_vocab_size,
            token_type_embedding_name="token_type_embeddings",
            use_position_embeddings=True,
            position_embedding_name="position_embeddings",
            initializer_range=config.initializer_range,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob)
        return self.embedding_output


class MainTransformer(TransformerBase):
    def __init__(self, config, ssdr_config, input_ids, input_mask, segment_ids, use_one_hot_embeddings):
        super(MainTransformer, self).__init__(config, input_ids, input_mask, segment_ids, use_one_hot_embeddings)
        self.layers_before_key_pooling = 3 #
        self.all_layer_outputs = []
        self.last_key_layer = None
        self.key_dimension = ssdr_config.key_dimension
        self.num_merge_layers = ssdr_config.num_merge_layers
        self.max_loc_length = ssdr_config.max_loc_length
        self.key_pooling = {
            "max_pooling": self.max_pooling,
            "mean_pooling": self.mean_pooling,
            "last_pooling": self.last_pooling,

        } [ssdr_config.key_pooling_method ]

    def max_pooling(self, key_vectors):
        return tf.math.reduce_max(key_vectors, 1)

    def mean_pooling(self, key_vectors):
        num_tokens = tf.expand_dims(tf.cast(tf.reduce_sum(self.input_mask, axis=1), tf.float32), 1)
        key_sum = tf.reduce_sum(key_vectors, axis=1)  # [batch_size, mr_num_route]
        key_avg = tf.math.divide(key_sum, num_tokens)
        return key_avg

    def last_pooling(self, key_vectors):
        return key_vectors[:, -1, :]

    def build_key(self):
        with tf.compat.v1.variable_scope("embeddings"):
            input_tensor = self.get_embeddings(self.input_ids, self.segment_ids)
            self.input_shape = bc.get_shape_list(input_tensor, expected_rank=3)

        with tf.compat.v1.variable_scope("encoder"):
            self.attention_mask = bc.create_attention_mask_from_input_mask(
                input_tensor, self.input_mask)
            prev_output = bc.reshape_to_matrix(input_tensor)
            for layer_idx in range(self.layers_before_key_pooling):
                with tf.compat.v1.variable_scope("layer_%d" % layer_idx):
                    intermediate_output, prev_output = self.forward_layer(prev_output)

                    final_output = bc.reshape_from_matrix(prev_output, self.input_shape)
                    self.all_layer_outputs.append(final_output)

        self.last_key_layer = prev_output
        with tf.compat.v1.variable_scope("mr_key"):
            key_vectors = bc.dense(self.key_dimension, self.initializer)(intermediate_output)
            key_vectors = tf.reshape(key_vectors, [self.batch_size, self.seq_length, self.key_dimension])
            key_output = self.key_pooling(key_vectors)
        return key_output

    # value_out : [batch, n_layers, hidden_dims]
    def build_remain(self, value_out, locations):
        def select_align_value(value_out, merge_layer_idx, flat_location):
            value_add_at_layer = value_out[:, merge_layer_idx, :]
            value_add_at_layer = tf.reshape(tf.tile(value_add_at_layer, [1, self.max_loc_length]),
                                            [-1, self.config.hidden_size])
            valid_mask = tf.cast(tf.not_equal(flat_location, 0), tf.float32)
            value_add_at_layer = value_add_at_layer * valid_mask
            return value_add_at_layer

        with tf.compat.v1.variable_scope("encoder"):
            n_remaining_layers = self.config.num_hidden_layers - self.layers_before_key_pooling

            # location : [batch_size, max_locations)
            offset = tf.expand_dims(tf.range(self.batch_size, dtype=tf.int32) * self.seq_length, 1)
            flat_location = tf.cast(locations, tf.int32) + offset
            flat_location = tf.reshape(flat_location, [-1, 1]) #[batch_size*max_location, 1])

            l_begin = self.layers_before_key_pooling
            l_end = self.layers_before_key_pooling +  self.num_merge_layers

            prev_output = self.last_key_layer
            for layer_idx in range(l_begin, l_end):
                merge_layer_idx = layer_idx - l_begin
                value_add_at_layer = select_align_value(value_out, merge_layer_idx, flat_location)
                # [ batch_size* max_location, hidden_size]
                with tf.compat.v1.variable_scope("layer_%d" % layer_idx):
                    intermediate_output, prev_output = \
                        self.forward_layer_with_added(prev_output, value_add_at_layer, flat_location)

                    final_output = bc.reshape_from_matrix(prev_output, self.input_shape)
                    self.all_layer_outputs.append(final_output)

            l_begin = l_end
            for layer_idx in range(l_begin, self.config.num_hidden_layers):
                with tf.compat.v1.variable_scope("layer_%d" % layer_idx):
                    intermediate_output, prev_output = self.forward_layer(prev_output)
                    final_output = bc.reshape_from_matrix(prev_output, self.input_shape)
                    self.all_layer_outputs.append(final_output)

        return self.all_layer_outputs


    def forward_layer_with_added(self, prev_output, added_value, locations):
        hidden_size = self.config.hidden_size
        layer_input = prev_output
        attention_output = self_attention_with_add(layer_input,
                                             self.attention_mask,
                                             self.config,
                                             self.batch_size,
                                             self.seq_length,
                                             hidden_size,
                                             self.initializer,
                                             added_value, locations)

        with tf.compat.v1.variable_scope("intermediate"):
            intermediate_output = bc.dense(self.config.intermediate_size, self.initializer,
                                           activation=bc.get_activation(self.config.hidden_act))(attention_output)

        with tf.compat.v1.variable_scope("output"):
            layer_output = bc.dense(hidden_size, self.initializer)(intermediate_output)
            layer_output = bc.dropout(layer_output, self.config.hidden_dropout_prob)
            layer_output = bc.layer_norm(layer_output + attention_output)
            prev_output = layer_output
        return intermediate_output, layer_output

def self_attention_with_add(layer_input,
                            attention_mask,
                            config,
                            batch_size,
                            seq_length,
                            hidden_size,
                            initializer,
                            values,
                            add_locations
                            ):

    attention_head_size = int(hidden_size / config.num_attention_heads)
    with tf.compat.v1.variable_scope("attention"):
        attention_heads = []
        with tf.compat.v1.variable_scope("self"):
            attention_head = bc.attention_layer(
                from_tensor=layer_input,
                to_tensor=layer_input,
                attention_mask=attention_mask,
                num_attention_heads=config.num_attention_heads,
                size_per_head=attention_head_size,
                attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                initializer_range=config.initializer_range,
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

        # [batch*seq_length, hidden_dim] , [batch, n_locations]
        attention_output = tf.tensor_scatter_nd_add(attention_output, add_locations, values)

        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        with tf.compat.v1.variable_scope("output"):
            attention_output = bc.dense(hidden_size, initializer)(attention_output)
            attention_output = bc.dropout(attention_output, config.hidden_dropout_prob)
            attention_output = bc.layer_norm(attention_output + layer_input)
    return attention_output


class SecondTransformer(TransformerBase):
    def __init__(self, config, ssdr_config, input_ids, input_mask, segment_ids, use_one_hot_embeddings):
        super(SecondTransformer, self).__init__(config, input_ids, input_mask, segment_ids, use_one_hot_embeddings)
        self.all_layer_outputs = []
        self.ssdr_config = ssdr_config

    def build(self, key):
        with tf.compat.v1.variable_scope("dict"):
            return self.build_by_attention(key)



    def build_by_attention(self, key):
        hidden_size = self.config.hidden_size
        with tf.compat.v1.variable_scope("embeddings"):
            input_tensor = self.get_embeddings(self.input_ids, self.segment_ids)

        with tf.compat.v1.variable_scope("encoder"):
            num_key_tokens = self.ssdr_config.num_key_tokens
            project_dim = hidden_size * num_key_tokens
            raw_key = bc.dense(project_dim, self.initializer)(key)
            key_tokens = tf.reshape(raw_key, [self.batch_size, num_key_tokens, hidden_size])

            input_tensor = tf.concat([key_tokens, input_tensor], axis=1)
            input_shape = bc.get_shape_list(input_tensor, expected_rank=3)

            mask_for_key = tf.ones([self.batch_size, num_key_tokens], dtype=tf.int64)
            self.input_mask = tf.concat([mask_for_key, self.input_mask], axis=1)
            self.seq_length = self.seq_length + num_key_tokens

            self.attention_mask = bc.create_attention_mask_from_input_mask(
                input_tensor, self.input_mask)
            prev_output = bc.reshape_to_matrix(input_tensor)
            for layer_idx in range(self.ssdr_config.num_hidden_layers):
                with tf.compat.v1.variable_scope("layer_%d" % layer_idx):
                    intermediate_output, prev_output = self.forward_layer(prev_output)
                    self.all_layer_outputs.append(prev_output)

            final_output = bc.reshape_from_matrix(prev_output, input_shape)
            self.scores = bc.dense(1, self.initializer)(final_output[:,0,:])

            if self.ssdr_config.info_pooling_method == "first_tokens":
                self.info_output = final_output[:, :num_key_tokens, :]
            elif self.ssdr_config.info_pooling_method == "max_pooling":
                self.info_output = tf.reduce_max(final_output, axis=1)

        return self.scores, self.info_output


def align_keys(keys, ab_mapping):
    return tf.gather(keys, ab_mapping)


def select_value(a_size, ab_mapping, b_scores, b_items, method):
    # [b_size]
    b_scores = tf.reshape(b_scores, [-1])
    b_size = bc.get_shape_list2(b_items)[0]
    t = tf.reshape(ab_mapping, [-1])
    t = tf.cast(t, tf.int32)
    indice = tf.stack([tf.range(b_size), t], 1)
    collect_bin = tf.scatter_nd(indice, tf.ones([b_size], tf.float32), [b_size, a_size])
    scattered_score = tf.transpose(tf.expand_dims(b_scores, 1) * collect_bin)
    # scattered_score :  [a_size, b_size], if not corresponding item, the score is zero

    if method == "max":
        selected_idx = tf.argmax(scattered_score, axis=1)

    elif method == "sample":
        remover = tf.transpose(tf.ones([b_size, a_size]) - collect_bin) * -10000.00
        scattered_score += remover
        selected_idx = categorical_sampling(scattered_score)
        #selected_idx = tf.random.categorical(scattered_score, 1)

    result = gather(b_items, selected_idx)
    #[n_items, n_layers, hidden]
    return result
    #return tf.gather_nd(b_items, selected_idx)

