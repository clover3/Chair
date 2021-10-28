from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.keras_model.bert_keras.bert_common_eager import get_shape_list_no_name
from models.transformer.bert_common_v2 import *
from models.transformer.bert_common_v2 import create_initializer
from tlm.model.base import BertModelInterface, transformer_model, BertModel
from tlm.qtype.qtype_embeddings import QTypeEmbeddingEmbPred, QTypeEmbeddingEmbPredDirect, QTypeEmbeddingWeightPred


def shift_construct(drop_embedding_output,
                    qtype_embedding,
                    qtype_len,
                    drop_input_ids,
                    drop_input_mask,
                    drop_token_type_ids,
                    sep_id
                    ):
    # [CLS] [QType Emb] [Content tokens] [SEP] [Document tokens] [SEP]
    input_shape = get_shape_list_no_name(drop_embedding_output)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    hidden_size = input_shape[2]
    is_seg1 = tf.equal(drop_token_type_ids, 0)
    is_seg2 = tf.equal(drop_token_type_ids, 1)
    is_sep = tf.equal(drop_input_ids, sep_id)
    shift_right = tf.logical_or(is_sep, is_seg2)
    shift_right_mask = tf.expand_dims(tf.cast(shift_right, tf.float32), 2)
    embedding_to_shift = shift_right_mask * drop_embedding_output
    right = embedding_to_shift[:, :-qtype_len, :]
    left = tf.zeros([batch_size, qtype_len, hidden_size], tf.float32)
    shifted_embedding_right = tf.concat([left, right], axis=1)

    cls = drop_embedding_output[:, :1, :]
    left_remain = tf.logical_and(is_seg1, tf.logical_not(is_sep))
    left_remain_mask = tf.expand_dims(tf.cast(left_remain, tf.float32), 2)
    embedding_content_tokens_and_cls = left_remain_mask * drop_embedding_output
    embedding_content_tokens = embedding_content_tokens_and_cls[:, 1:-qtype_len, :]
    shifted_embedding_left = tf.concat([cls, qtype_embedding, embedding_content_tokens], axis=1)

    shifted_embedding = shifted_embedding_left + shifted_embedding_right

    # input_mask
    input_mask_cls = drop_input_mask[:, :1, ]
    input_mask_qtype = tf.ones([batch_size, qtype_len], tf.int32)
    input_mask_remain = drop_input_mask[:, 1:-qtype_len, ]
    shifted_input_mask = tf.concat([input_mask_cls, input_mask_qtype, input_mask_remain], axis=1)

    # segment_ids
    segment_mask_cls = drop_token_type_ids[:, :1, ]
    segment_mask_qtype = tf.zeros([batch_size, qtype_len], tf.int32)
    segment_mask_remain = drop_token_type_ids[:, 1:-qtype_len, ]
    shifted_segment_ids = tf.concat([segment_mask_cls, segment_mask_qtype, segment_mask_remain], axis=1)

    return shifted_embedding, shifted_input_mask, shifted_segment_ids


# How to encode [seq_length, 13, 728]
# 13 * 728 ~= 10^4


class MLP(tf.keras.layers.Layer):
    def __init__(self, config, hidden_dims):
        super(MLP, self).__init__()
        self.l_list = []
        initializer = create_initializer(config.initializer_range)
        for d in hidden_dims:
            l = tf.keras.layers.Dense(d, activation=gelu, kernel_initializer=initializer)
            self.l_list.append(l)

    # input_shape = [batch_size, seq_length, 13, 728]
    def call(self, inputs, **kwargs):
        input_shape = get_shape_list(inputs)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        h = tf.reshape(inputs, [batch_size, seq_length, -1])
        for dense_layer in self.l_list:
            h = dense_layer(h)
        return h


class Residual(tf.keras.layers.Layer):
    def __init__(self, config):
        super(Residual, self).__init__()
        initializer = create_initializer(config.initializer_range)
        def dense(activation=None):
            return tf.keras.layers.Dense(config.hidden_size,
                                         kernel_initializer=initializer,
                                         activation=activation)
        n = config.num_hidden_layers
        self.num_hidden_layers = config.num_hidden_layers

        self.projection_dense_list = [dense(activation=gelu) for _ in range(n)]
        self.transform_dense_1 = [dense(activation=gelu) for _ in range(n)]
        self.transform_dense_2 = [dense() for _ in range(n)]

        def layer_norm():
            return tf.keras.layers.LayerNormalization(epsilon=1e-3, axis=-1)

        self.layer_norm_list_1 = [layer_norm() for _ in range(n)]
        self.layer_norm_list_2 = [layer_norm() for _ in range(n)]
        self.dropout_prob = config.hidden_dropout_prob

    # input_shape = [batch_size, seq_length, 728 * 13]
    def call(self, inputs, **kwargs):
        h = inputs
        embedding_layer_tensor = h[:, :, 0, :]
        prev_output = embedding_layer_tensor
        for layer_no in range(self.num_hidden_layers):
            cur_layer = h[:, :, 1 + layer_no, :]
            layer_projection = self.projection_dense_list[layer_no](cur_layer)
            layer_projection = tf.nn.dropout(layer_projection, rate=self.dropout_prob)
            added_rep = self.layer_norm_list_1[layer_no](prev_output + layer_projection)
            intermediate = self.transform_dense_1[layer_no](added_rep)
            layer_vector = self.transform_dense_2[layer_no](intermediate)
            layer_vector = tf.nn.dropout(layer_vector, rate=self.dropout_prob)
            cur_output = self.layer_norm_list_2[layer_no](prev_output + layer_vector)
            prev_output = cur_output

        # reduce seq_length axis
        h = tf.reduce_sum(prev_output, axis=1)
        return h # [batch_size, 728]


class BertQType(BertModelInterface):
    def __init__(self,
                 ):
        super(BertQType, self).__init__()

    def build_tower2(self, model_config, model_config_predict, all_layers_seg1, drop_input_ids, drop_input_mask, drop_token_type_ids, is_training,
                     sep_id, use_one_hot_embeddings):
        embedding_modeling, inner_modeling = model_config.q_modeling.split("_")
        q_modeling_class = {
            'QTypeEmbeddingEmbPred': QTypeEmbeddingEmbPred,
            'QTypeEmbeddingEmbPredDirect': QTypeEmbeddingEmbPredDirect,
            'QTypeEmbeddingWeightPred': QTypeEmbeddingWeightPred,
        }[embedding_modeling]
        s = model_config.hidden_size
        inner_layer: tf.keras.layers.Layer = {
            'MLP': MLP(model_config, [s, s]),
            'residual': Residual(model_config),
        }[inner_modeling]
        q_embedding_model: tf.keras.layers.Layer = q_modeling_class(model_config, inner_layer)
        qtype_embeddings = q_embedding_model(all_layers_seg1)  # [batch_size, type_len, hidden_dim]
        input_shape = get_shape_list(drop_input_ids, expected_rank=2)
        with tf.compat.v1.variable_scope("bert"):
            with tf.compat.v1.variable_scope("embeddings"):
                # Perform embedding lookup on the word ids.
                (drop_embedding_output, self.drop_embedding_table) = embedding_lookup(
                    input_ids=drop_input_ids,
                    vocab_size=model_config_predict.vocab_size,
                    embedding_size=model_config_predict.hidden_size,
                    initializer_range=model_config_predict.initializer_range,
                    word_embedding_name="word_embeddings",
                    use_one_hot_embeddings=use_one_hot_embeddings)

                shifted_embedding, shifted_input_mask, shifted_segment_ids \
                    = shift_construct(drop_embedding_output, qtype_embeddings, model_config.qtype_len,
                                      drop_input_ids, drop_input_mask, drop_token_type_ids, sep_id)

                # Add positional embeddings and token type embeddings, then layer
                # normalize and perform dropout.
                drop_embedding_output = embedding_postprocessor(
                    input_tensor=shifted_embedding,
                    use_token_type=True,
                    token_type_ids=shifted_segment_ids,
                    token_type_vocab_size=model_config_predict.type_vocab_size,
                    token_type_embedding_name="token_type_embeddings",
                    use_position_embeddings=True,
                    position_embedding_name="position_embeddings",
                    initializer_range=model_config_predict.initializer_range,
                    max_position_embeddings=model_config_predict.max_position_embeddings,
                    dropout_prob=model_config_predict.hidden_dropout_prob)

            with tf.compat.v1.variable_scope("encoder"):
                drop_attention_mask = create_attention_mask_from_input_mask(
                    drop_embedding_output, shifted_input_mask)

                self.drop_all_encoder_layers = transformer_model(
                    input_tensor=drop_embedding_output,
                    attention_mask=drop_attention_mask,
                    input_mask=shifted_input_mask,
                    hidden_size=model_config_predict.hidden_size,
                    num_hidden_layers=model_config_predict.num_hidden_layers,
                    num_attention_heads=model_config_predict.num_attention_heads,
                    is_training=is_training,
                    intermediate_size=model_config_predict.intermediate_size,
                    intermediate_act_fn=get_activation(model_config_predict.hidden_act),
                    hidden_dropout_prob=model_config_predict.hidden_dropout_prob,
                    attention_probs_dropout_prob=model_config_predict.attention_probs_dropout_prob,
                    initializer_range=model_config_predict.initializer_range,
                    do_return_all_layers=True)

            drop_sequence_output = self.drop_all_encoder_layers[-1]
            with tf.compat.v1.variable_scope("pooler"):
                first_token_tensor = tf.squeeze(drop_sequence_output[:, 0:1, :], axis=1)
                drop_pooled_output = tf.keras.layers.Dense(model_config_predict.hidden_size,
                                                           activation=tf.keras.activations.tanh,
                                                           kernel_initializer=create_initializer(
                                                               model_config_predict.initializer_range))(
                    first_token_tensor)

                self.drop_pooled_output = drop_pooled_output

    def build_tower1(self, config, is_training, orig_input_ids, orig_input_mask, orig_token_type_ids,
                     use_one_hot_embeddings):
        input_shape = get_shape_list(orig_input_ids, expected_rank=2)
        batch_size = input_shape[0]
        model_a = BertModel(config, is_training, orig_input_ids, orig_input_mask, orig_token_type_ids,
                            use_one_hot_embeddings)
        emb_layer = model_a.get_embedding_output()
        layers = model_a.get_all_encoder_layers()

        all_layers = tf.stack([emb_layer] + layers, 2)  # [batch_size, seq_length, num_layer+1, hidden_dim]
        is_seg1 = tf.equal(orig_token_type_ids, 0)
        is_seg1_mask = tf.cast(tf.reshape(is_seg1, [batch_size, -1, 1, 1]), tf.float32)
        all_layers_seg1 = all_layers * is_seg1_mask
        maybe_q_len_enough = 64
        all_layers_seg1 = all_layers_seg1[:, :maybe_q_len_enough, :, :]
        num_layer_plus_one = config.num_hidden_layers + 1
        dim_per_token = num_layer_plus_one * config.hidden_size
        flatten_all_layers = tf.reshape(all_layers_seg1, [batch_size, -1, dim_per_token])
        self.pooled_output = model_a.pooled_output
        self.all_encoder_layers = model_a.all_encoder_layers
        return all_layers_seg1