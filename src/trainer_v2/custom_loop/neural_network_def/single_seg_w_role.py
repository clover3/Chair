from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

from models.transformer.bert_common_v2 import get_shape_list2
from trainer_v2.bert_for_tf2 import BertModelLayer
from trainer_v2.custom_loop.modeling_common.bert_common import define_bert_input, load_bert_checkpoint, \
    load_stock_weights, load_pooler, load_stock_weights_bert_like
from trainer_v2.custom_loop.neural_network_def.inner_network import ClassificationModelIF
from trainer_v2.custom_loop.neural_network_def.var_local_decisions import NLC, transform_inputs_for_ts, keep_seg_12, \
    TransformInputsForTS

import params_flow as pf
from bert.embeddings import BertEmbeddingsLayer
from bert.layer import Layer
from tensorflow import keras

from trainer_v2.bert_for_tf2.transformer import TransformerEncoderLayer


class RoleInsertedBERT(Layer):
    class Params(BertEmbeddingsLayer.Params,
                 TransformerEncoderLayer.Params):
        pass

    # noinspection PyUnusedLocal
    def _construct(self, **kwargs):
        super()._construct(**kwargs)
        self.embeddings_layer = BertEmbeddingsLayer.from_params(
            self.params,
            name="embeddings"
        )
        # create all transformer encoder sub-layers
        self.encoders_layer = TransformerEncoderLayer.from_params(
            self.params,
            name="encoder"
        )
        self.support_masking = True

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        if isinstance(input_shape, list):
            assert len(input_shape) == 3
            input_ids_shape, token_type_ids_shape, vector_shape = input_shape
            self.input_spec = [keras.layers.InputSpec(shape=input_ids_shape),
                               keras.layers.InputSpec(shape=token_type_ids_shape),
                               keras.layers.InputSpec(shape=vector_shape)
                               ]
        else:
            input_ids_shape = input_shape
            self.input_spec = keras.layers.InputSpec(shape=input_ids_shape)
        super(RoleInsertedBERT, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            assert len(input_shape) == 3
            input_ids_shape, _, _ = input_shape
        else:
            input_ids_shape = input_shape

        output_shape = list(input_ids_shape) + [self.params.hidden_size]
        return output_shape

    def apply_adapter_freeze(self):
        """ Should be called once the model has been built to freeze
        all bet the adapter and layer normalization layers in BERT.
        """
        if self.params.adapter_size is not None:
            def freeze_selector(layer):
                return layer.name not in ["adapter-up", "adapter-down", "LayerNorm", "extra_word_embeddings"]
            pf.utils.freeze_leaf_layers(self, freeze_selector)

    def call(self, inputs_all, mask=None, training=None):
        input_ids, segment_ids, role_vector = inputs_all
        inputs = [input_ids, segment_ids]
        if mask is None:
            mask = self.embeddings_layer.compute_mask(inputs)

        embedding_output = self.embeddings_layer(inputs, mask=mask, training=training)
        embedding_output = embedding_output + role_vector
        output           = self.encoders_layer(embedding_output, mask=mask, training=training)
        return output   # [B, seq_len, hidden_size]


class RoleLayer1(tf.keras.layers.Layer):
    def __init__(self, hidden_size):
        super(RoleLayer1, self).__init__()
        self.k = 10
        self.project_dim = hidden_size * 4

        Dense = tf.keras.layers.Dense
        self.projector = Dense(self.project_dim, activation='relu', name="project")
        self.l2 = Dense(self.k, activation='softmax')
        init_val = tf.random_normal_initializer(0.01)([self.k, hidden_size])
        self.role_vectors = tf.Variable(init_val, trainable=True)

    def call(self, inputs, *args, **kwargs):
        # B: batch size
        # L: Sequence length
        # H: Hidden size
        # Return:  [2*B, L, H]
        h_encoded, h_input_ids, h_token_type_ids = inputs
        h = self.projector(h_encoded)
        role_probs = self.l2(h)  # [B, L, K]
        role_vector = tf.tensordot(role_probs, self.role_vectors, [2, 0])  # [B, L, H]
        return tf.concat([role_vector, role_vector], axis=0)


class SingleSegWRole(ClassificationModelIF):
    def __init__(self, combine_ld):
        super(SingleSegWRole, self).__init__()
        self.combine_ld = combine_ld

    def build_model(self, bert_params, config: NLC):
        role_insert_bert = RoleInsertedBERT.from_params(bert_params, name="ph_enc/bert")
        pooler = tf.keras.layers.Dense(bert_params.hidden_size, activation=tf.nn.tanh,
                                       name="ph_enc/bert/pooler/dense")

        role_encoder = BertModelLayer.from_params(bert_params, name="h_enc/bert")

        num_classes = config.num_classes
        num_local_classes = config.num_local_classes
        max_seq_length = config.max_seq_length
        input_ids, token_type_ids = define_bert_input(max_seq_length, "")  # [B, L]

        ph_input_ids, ph_segment_ids = TransformInputsForTS()([input_ids, token_type_ids])  # [2*B, L]
        h_input_ids, h_token_type_ids = keep_seg_12(input_ids, token_type_ids)  # [B, L]

        h_token_type_ids0 = tf.minimum(h_token_type_ids, 1)
        h_encoded = role_encoder([h_input_ids, h_token_type_ids0])  # [B, L]
        role_layer = RoleLayer1(bert_params.hidden_size)
        role_vector = role_layer([h_encoded, h_input_ids, h_token_type_ids])  # [2*B, L, H]
        ph_inputs = [ph_input_ids, ph_segment_ids, role_vector]
        encoded_output = role_insert_bert(ph_inputs)
        feature_rep_flat = pooler(encoded_output[:, 0, :])

        B, _ = get_shape_list2(input_ids)
        feature_rep = tf.reshape(feature_rep_flat, [2, B, bert_params.hidden_size], "feature_rep_flat")
        feature_rep = tf.transpose(feature_rep, [1, 0, 2])
        # [batch_size, num_window, dim2 ]
        hidden = tf.keras.layers.Dense(bert_params.hidden_size, activation='relu')(feature_rep)
        local_decisions = tf.keras.layers.Dense(num_local_classes, activation=tf.nn.softmax)(hidden)
        comb_layer = self.combine_ld()
        output = comb_layer(local_decisions)
        inputs = (input_ids, token_type_ids)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        self.model: keras.Model = model
        self.bert_l_list = [role_insert_bert, role_encoder]
        self.pooler_list = [pooler]

    def get_keras_model(self):
        return self.model

    def init_checkpoint(self, init_checkpoint):
        for l_bert in self.bert_l_list:
            load_stock_weights_bert_like(l_bert, init_checkpoint, n_expected_restore=197)
        for pooler in self.pooler_list:
            load_pooler(pooler, init_checkpoint)

