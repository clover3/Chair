

import copy

import tensorflow as tf

import models.transformer.bert_common_v2 as bc
import tlm.model.base as base
from tlm.dictionary.sense_selecting_dictionary_reader import TransformerBase, select_value, \
    get_pooler


class ERDR(base.BertModelInterface):
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
                 ab_mapping_mask=None,
                 use_one_hot_embeddings=True,
                 scope=None,
                 ):
        super(ERDR, self).__init__()

        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        batch_size, seq_length = bc.get_batch_and_seq_length(input_ids, 2)

        if input_mask is None:
            input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

        if d_segment_ids is None:
            d_segment_ids = d_input_mask

        with tf.compat.v1.variable_scope(scope, default_name="bert"):
            self.main_transformer = MainTransformer(
                config, ssdr_config, input_ids, input_mask, token_type_ids, use_one_hot_embeddings)

            self.dict_tranformer = SimpleTransformer(
                config, ssdr_config, d_input_ids, d_input_mask, d_segment_ids, use_one_hot_embeddings)

            scores, info_vectors = self.dict_tranformer.build()
            self.scores = scores
            info_vector = select_value(batch_size, ab_mapping, scores, info_vectors, "sample", ab_mapping_mask)
            all_encoder_layers = self.main_transformer.build(info_vector, d_location_ids)

            self.all_encoder_layers = all_encoder_layers
            self.sequence_output = self.all_encoder_layers[-1]
            self.pooled_output = get_pooler(self.sequence_output, config)
            self.embedding_output = self.main_transformer.embedding_output
            self.embedding_table = self.main_transformer.embedding_table


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

    def build(self, value_out, locations):
        with tf.compat.v1.variable_scope("embeddings"):
            input_tensor = self.get_embeddings(self.input_ids, self.segment_ids)
            self.input_shape = bc.get_shape_list(input_tensor, expected_rank=3)

        with tf.compat.v1.variable_scope("encoder"):
            self.attention_mask = bc.create_attention_mask_from_input_mask(
                input_tensor, self.input_mask)
            prev_output = bc.reshape_to_matrix(input_tensor)
            prev_output = tf.tensor_scatter_nd_update(prev_output, locations, value_out)

            for layer_idx in range(self.config.num_hidden_layers):
                with tf.compat.v1.variable_scope("layer_%d" % layer_idx):
                    intermediate_output, prev_output = self.forward_layer(prev_output)
                    final_output = bc.reshape_from_matrix(prev_output, self.input_shape)
                    self.all_layer_outputs.append(final_output)

        return self.all_layer_outputs


class SimpleTransformer(TransformerBase):
    def __init__(self, config, ssdr_config, input_ids, input_mask, segment_ids, use_one_hot_embeddings):
        super(SimpleTransformer, self).__init__(config, input_ids, input_mask, segment_ids, use_one_hot_embeddings)
        self.all_layer_outputs = []
        self.ssdr_config = ssdr_config

    def build(self):
        with tf.compat.v1.variable_scope("dict"):
            with tf.compat.v1.variable_scope("embeddings"):
                input_tensor = self.get_embeddings(self.input_ids, self.segment_ids)

            with tf.compat.v1.variable_scope("encoder"):
                num_key_tokens = self.ssdr_config.num_key_tokens
                input_shape = bc.get_shape_list(input_tensor, expected_rank=3)

                mask_for_key = tf.ones([self.batch_size, num_key_tokens], dtype=tf.int64)
                self.input_mask = tf.cast(self.input_mask, tf.int64)
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


