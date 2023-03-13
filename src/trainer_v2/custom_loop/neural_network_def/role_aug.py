import tensorflow as tf
from bert.embeddings import BertEmbeddingsLayer
from tensorflow import keras

from models.transformer.bert_common_v2 import get_shape_list2
from trainer_v2.bert_for_tf2 import BertModelLayer
from trainer_v2.bert_for_tf2.w_mask.transformer import TransformerEncoderLayerWMask
from trainer_v2.custom_loop.modeling_common.bert_common import load_stock_weights, define_bert_input, \
    load_stock_weights_encoder_only, load_stock_weights_embedding
from trainer_v2.custom_loop.modeling_common.network_utils import VectorThreeFeature, \
    MeanProjectionEnc, TwoLayerDense, ChunkAttentionMaskLayer
from trainer_v2.custom_loop.neural_network_def.inner_network import BertBasedModelIF
from trainer_v2.custom_loop.neural_network_def.siamese import ModelConfig2SegProject


class DummyZeroLayer(tf.keras.layers.Layer):
    def __init__(self, out_dimension):
        super(DummyZeroLayer, self).__init__()
        self.out_dimension = out_dimension
        self.out_shape = None

    def build(self, input_shape):
        if isinstance(input_shape, list):
            assert False
        else:
            self.input_spec = keras.layers.InputSpec(shape=input_shape)
            self.out_shape = tf.TensorShape(input_shape)[:2] + [self.out_dimension]
        super(DummyZeroLayer, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):
        B, L, _ = get_shape_list2(inputs)
        return tf.zeros([B, L, self.out_dimension])


class ChunkStartRole(BertBasedModelIF):
    def __init__(self,
                 combine_local_decisions_layer,
                 lower_project_layer,
                 ):
        super(ChunkStartRole, self).__init__()
        self.combine_local_decisions_layer = combine_local_decisions_layer
        self.lower_project_layer = lower_project_layer

    def build_model(self, bert_params, config: ModelConfig2SegProject):
        # Parameters
        upper_proj_dim = config.project_dim
        num_classes = config.num_classes
        max_seq_len1 = config.max_seq_length1
        max_seq_len2 = config.max_seq_length2

        # Layers
        Dense = tf.keras.layers.Dense
        encoder1 = MeanProjectionEnc(bert_params, upper_proj_dim, "encoder1")
        encoder2_lower = BertModelLayer.from_params(bert_params, name="encoder2_lower/bert")
        encoder2_upper_embeddings_layer = BertEmbeddingsLayer.from_params(bert_params, name="encoder_upper/embeddings")
        encoder2_upper = TransformerEncoderLayerWMask.from_params(bert_params, name="encoder_upper/encoder")
        encoder2_project = self.lower_project_layer(bert_params.hidden_size)
        project2_upper = Dense(upper_proj_dim, activation='relu')
        combine_local_decisions = self.combine_local_decisions_layer()

        # Inputs
        l_input_ids1, l_token_type_ids1 = define_bert_input(max_seq_len1, "1")
        l_input_ids2, l_token_type_ids2 = define_bert_input(max_seq_len2, "2")
        input_mask1 = tf.cast(tf.not_equal(l_input_ids1, 0), tf.int32)
        input_mask2 = tf.cast(tf.not_equal(l_input_ids2, 0), tf.int32)
        dummy_token_type_ids1 = tf.zeros_like(l_token_type_ids1, tf.int32)
        dummy_token_type_ids2 = tf.zeros_like(l_token_type_ids2, tf.int32)
        chunk_start_mask2 = l_token_type_ids2
        input_mask_concat = tf.concat([input_mask1, input_mask2], axis=1)
        # Encoder 1
        rep1 = encoder1([l_input_ids1, dummy_token_type_ids1])
        rep1_ = tf.expand_dims(rep1, 1)
        rep1_stacked = tf.tile(rep1_, [1, max_seq_len2, 1])

        # Encoder 2
        attention_mask = ChunkAttentionMaskLayer()(chunk_start_mask2)
        lower_seq_out = encoder2_lower([l_input_ids2, dummy_token_type_ids2])
        lower_output = encoder2_project(lower_seq_out)

        embedding_output = encoder2_upper_embeddings_layer([l_input_ids2, dummy_token_type_ids2],
                                                           mask=input_mask2)
        embedding_output = embedding_output + lower_output
        upper_seq_out = encoder2_upper(embedding_output, attention_mask=attention_mask)
        rep2_stacked = project2_upper(upper_seq_out)

        # Build local decisions 2
        vtf = VectorThreeFeature()
        feature_rep = vtf((rep1_stacked, rep2_stacked))

        local_decisions = TwoLayerDense(bert_params.hidden_size, num_classes,
                                        name="local_decision")(feature_rep)
        is_valid_decision = chunk_start_mask2 * input_mask2
        output = combine_local_decisions([local_decisions, is_valid_decision])

        # Define models
        inputs = (l_input_ids1, l_token_type_ids1, l_input_ids2, l_token_type_ids2)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        self.model: keras.Model = model
        self.encoder2_project = encoder2_project
        self.l_bert_list = [encoder1.l_bert, encoder2_lower]
        self.l_transformer = encoder2_upper
        self.upper_embedding = encoder2_upper_embeddings_layer

    def get_keras_model(self):
        return self.model

    def init_checkpoint(self, init_checkpoint):
        for l_bert in self.l_bert_list:
            load_stock_weights(l_bert, init_checkpoint, n_expected_restore=197)
        n_embedding_vars = 5
        load_stock_weights_encoder_only(self.l_transformer, init_checkpoint, n_expected_restore=197 -n_embedding_vars)
        load_stock_weights_embedding(self.upper_embedding, init_checkpoint, n_expected_restore=n_embedding_vars)