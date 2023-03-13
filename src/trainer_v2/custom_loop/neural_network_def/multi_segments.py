import tensorflow as tf
from tensorflow import keras

from models.transformer.bert_common_v2 import get_shape_list2
from trainer_v2.bert_for_tf2 import BertModelLayer
from trainer_v2.bert_for_tf2.spit_attn_probs.bert_layer import BertModelLayerSAP
from trainer_v2.bert_for_tf2.w_mask.model_bert_mask import BertModelLayerWMask, load_stock_weights_for_wmask, \
    load_stock_weights_nc
from trainer_v2.custom_loop.modeling_common.bert_common import load_stock_weights
from trainer_v2.custom_loop.modeling_common.network_utils import VectorThreeFeature, \
    TwoLayerDense, ChunkAttentionMaskLayer
from trainer_v2.custom_loop.neural_network_def.inner_network import BertBasedModelIF
from trainer_v2.custom_loop.neural_network_def.siamese import ModelConfig2SegProject


# Projection + Mean+ Concat
class SingleSegment(BertBasedModelIF):
    def __init__(self, combine_local_decisions_layer):
        super(SingleSegment, self).__init__()
        self.combine_local_decisions_layer = combine_local_decisions_layer

    def build_model(self, bert_params, config: ModelConfig2SegProject):
        Dense = tf.keras.layers.Dense

        encoder1_lower = BertModelLayer.from_params(bert_params, name="encoder1/bert")
        projector1 = Dense(config.project_dim, activation='relu', name="encoder1/project")

        encoder2_lower = BertModelLayer.from_params(bert_params, name="encoder2/bert")
        projector2 = Dense(config.project_dim, activation='relu', name="encoder2/project")

        max_seq_len1 = config.max_seq_length1
        max_seq_len2 = config.max_seq_length2
        num_classes = config.num_classes

        l_input_ids1 = keras.layers.Input(shape=(max_seq_len1,), dtype='int32', name="input_ids1")
        l_token_type_ids1 = keras.layers.Input(shape=(max_seq_len1,), dtype='int32', name="segment_ids1")

        l_input_ids2 = keras.layers.Input(shape=(max_seq_len2,), dtype='int32', name="input_ids2")
        l_token_type_ids2 = keras.layers.Input(shape=(max_seq_len2,), dtype='int32', name="segment_ids2")

        seq_out1 = encoder1_lower([l_input_ids1, l_token_type_ids1])
        seq_out1_p = projector1(seq_out1)
        rep1 = tf.reduce_mean(seq_out1_p, axis=1)
        rep1_ = tf.expand_dims(rep1, 1)
        rep1_stacked = tf.tile(rep1_, [1, max_seq_len2, 1])

        seq_out2 = encoder2_lower([l_input_ids2, l_token_type_ids2])
        rep2_stacked = projector2(seq_out2)
        vtf = VectorThreeFeature()
        feature_rep = vtf((rep1_stacked, rep2_stacked))

        local_decisions = TwoLayerDense(bert_params.hidden_size, num_classes,
                                        name="local_decision")(feature_rep)
        B, _ = get_shape_list2(l_input_ids1)
        is_valid_decision_base = tf.concat([tf.ones([1, 1]), tf.zeros([1, max_seq_len2-1])], axis=1)
        is_valid_decision = tf.ones_like(l_input_ids2, tf.float32) * is_valid_decision_base
        self.is_valid_decision = is_valid_decision
        self.local_decisions = local_decisions
        combine_local_decisions = self.combine_local_decisions_layer()
        self.cld = combine_local_decisions
        output = combine_local_decisions([local_decisions, is_valid_decision])
        inputs = (l_input_ids1, l_token_type_ids1, l_input_ids2, l_token_type_ids2)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        self.model: keras.Model = model
        self.l_bert_list = [encoder1_lower, encoder2_lower]

    def get_keras_model(self):
        return self.model

    def init_checkpoint(self, init_checkpoint):
        l_bert1, l_bert2 = self.l_bert_list
        load_stock_weights(l_bert1, init_checkpoint, n_expected_restore=197)
        load_stock_weights(l_bert2, init_checkpoint, n_expected_restore=197)


def create_attention_mask(from_vector, input_mask):
    """
    Creates 3D attention.
    :param from_shape:  [batch_size, from_seq_len, ...]
    :param input_mask:  [batch_size, seq_len]
    :return: [batch_size, from_seq_len, seq_len]
    """
    from_shape = tf.shape(input=from_vector)
    mask = tf.cast(tf.expand_dims(input_mask, axis=1), tf.float32)                   # [B, 1, T]
    ones = tf.expand_dims(tf.ones(shape=from_shape[:2], dtype=tf.float32), axis=-1)  # [B, F, 1]
    mask = ones * mask  # broadcast along two dimensions
    return mask  # [B, F, T]


class SingleSegment2(BertBasedModelIF):
    def __init__(self, combine_local_decisions_layer):
        super(SingleSegment2, self).__init__()
        self.combine_local_decisions_layer = combine_local_decisions_layer

    def build_model(self, bert_params, config: ModelConfig2SegProject):
        Dense = tf.keras.layers.Dense

        encoder1_lower = BertModelLayerSAP.from_params(bert_params, name="encoder1/bert")
        projector1 = Dense(config.project_dim, activation='relu', name="encoder1/project")

        encoder2_lower = BertModelLayerSAP.from_params(bert_params, name="encoder2/bert")
        projector2 = Dense(config.project_dim, activation='relu', name="encoder2/project")

        max_seq_len1 = config.max_seq_length1
        max_seq_len2 = config.max_seq_length2
        num_classes = config.num_classes

        l_input_ids1 = keras.layers.Input(shape=(max_seq_len1,), dtype='int32', name="input_ids1")
        l_token_type_ids1 = keras.layers.Input(shape=(max_seq_len1,), dtype='int32', name="segment_ids1")
        input_mask1 = tf.cast(tf.not_equal(l_input_ids1, 0), tf.int32)

        l_input_ids2 = keras.layers.Input(shape=(max_seq_len2,), dtype='int32', name="input_ids2")
        l_token_type_ids2 = keras.layers.Input(shape=(max_seq_len2,), dtype='int32', name="segment_ids2")
        input_mask2 = tf.cast(tf.not_equal(l_input_ids2, 0), tf.int32)

        attention_mask1 = create_attention_mask(l_input_ids1, input_mask1)
        seq_out1, attn_probs1 = encoder1_lower([l_input_ids1, l_token_type_ids1], attention_mask=attention_mask1)
        seq_out1_p = projector1(seq_out1)
        rep1 = tf.reduce_mean(seq_out1_p, axis=1)
        rep1_ = tf.expand_dims(rep1, 1)
        rep1_stacked = tf.tile(rep1_, [1, max_seq_len2, 1])

        attention_mask2 = create_attention_mask(l_input_ids2, input_mask2)
        seq_out2, attn_probs2 = encoder2_lower([l_input_ids2, l_token_type_ids2], attention_mask=attention_mask2)
        rep2_stacked = projector2(seq_out2)
        vtf = VectorThreeFeature()
        feature_rep = vtf((rep1_stacked, rep2_stacked))

        local_decisions = TwoLayerDense(bert_params.hidden_size, num_classes,
                                        name="local_decision")(feature_rep)
        B, _ = get_shape_list2(l_input_ids1)
        is_valid_decision_base = tf.concat([tf.ones([1, 1]), tf.zeros([1, max_seq_len2-1])], axis=1)
        is_valid_decision = tf.ones_like(l_input_ids2, tf.float32) * is_valid_decision_base
        self.is_valid_decision = is_valid_decision
        self.local_decisions = local_decisions
        combine_local_decisions = self.combine_local_decisions_layer()
        self.cld = combine_local_decisions
        output = combine_local_decisions([local_decisions, is_valid_decision])
        inputs = (l_input_ids1, l_token_type_ids1, l_input_ids2, l_token_type_ids2)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        self.model: keras.Model = model
        self.l_bert_list = [encoder1_lower, encoder2_lower]

    def get_keras_model(self):
        return self.model

    def init_checkpoint(self, init_checkpoint):
        l_bert1, l_bert2 = self.l_bert_list
        load_stock_weights_nc(l_bert1, init_checkpoint, n_expected_restore=197)
        load_stock_weights_nc(l_bert2, init_checkpoint, n_expected_restore=197)


class TwoChunk(BertBasedModelIF):
    def __init__(self, combine_local_decisions_layer):
        super(TwoChunk, self).__init__()
        self.combine_local_decisions_layer = combine_local_decisions_layer

    def build_model(self, bert_params, config: ModelConfig2SegProject):
        Dense = tf.keras.layers.Dense

        encoder1 = BertModelLayer.from_params(bert_params, name="encoder1/bert")
        projector1 = Dense(config.project_dim, activation='relu', name="encoder1/project")

        encoder2 = BertModelLayerWMask.from_params(bert_params, name="encoder2/bert")
        projector2 = Dense(config.project_dim, activation='relu', name="encoder2/project")

        max_seq_len1 = config.max_seq_length1
        max_seq_len2 = config.max_seq_length2
        num_classes = config.num_classes

        l_input_ids1 = keras.layers.Input(shape=(max_seq_len1,), dtype='int32', name="input_ids1")
        l_token_type_ids1 = keras.layers.Input(shape=(max_seq_len1,), dtype='int32', name="segment_ids1")

        l_input_ids2 = keras.layers.Input(shape=(max_seq_len2,), dtype='int32', name="input_ids2")
        l_token_type_ids2 = keras.layers.Input(shape=(max_seq_len2,), dtype='int32', name="segment_ids2")

        seq_out1 = encoder1([l_input_ids1, l_token_type_ids1])
        seq_out1_p = projector1(seq_out1)
        rep1 = tf.reduce_mean(seq_out1_p, axis=1)
        rep1_ = tf.expand_dims(rep1, 1)
        rep1_stacked = tf.tile(rep1_, [1, max_seq_len2, 1])
        is_valid_decision_base = tf.concat([tf.ones([1, 1]), tf.zeros([1, max_seq_len2-1])], axis=1)

        first_chunk_len = 10
        second_chunk_len = max_seq_len2 - first_chunk_len
        chunk_start_mask2 = tf.concat([
            tf.ones([1, 1], tf.int32),
            tf.zeros([1, first_chunk_len-1], tf.int32),
            tf.ones([1, 1], tf.int32),
            tf.zeros([1, second_chunk_len - 1], tf.int32),
        ], axis=1)
        attention_mask = ChunkAttentionMaskLayer()(chunk_start_mask2)
        seq_out2 = encoder2([l_input_ids2, l_token_type_ids2], attention_mask=attention_mask)

        rep2_stacked = projector2(seq_out2)
        vtf = VectorThreeFeature()
        feature_rep = vtf((rep1_stacked, rep2_stacked))

        local_decisions = TwoLayerDense(bert_params.hidden_size, num_classes,
                                        name="local_decision")(feature_rep)
        B, _ = get_shape_list2(l_input_ids1)
        is_valid_decision = chunk_start_mask2
        self.is_valid_decision = is_valid_decision
        self.local_decisions = local_decisions
        combine_local_decisions = self.combine_local_decisions_layer()
        self.cld = combine_local_decisions
        output = combine_local_decisions([local_decisions, is_valid_decision])
        inputs = (l_input_ids1, l_token_type_ids1, l_input_ids2, l_token_type_ids2)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        self.model: keras.Model = model
        self.l_bert_list = [encoder1, encoder2]

    def get_keras_model(self):
        return self.model

    def init_checkpoint(self, init_checkpoint):
        l_bert1, l_bert2 = self.l_bert_list
        load_stock_weights(l_bert1, init_checkpoint, n_expected_restore=197)
        load_stock_weights_for_wmask(l_bert2, init_checkpoint, n_expected_restore=197)


class ChunkStartEncoder(BertBasedModelIF):
    # This expect segment_ids2 to indicate start of chunk
    # Each chunks can only attend to oneself.
    def __init__(self, combine_local_decisions_layer):
        super(ChunkStartEncoder, self).__init__()
        self.combine_local_decisions_layer = combine_local_decisions_layer

    def build_model(self, bert_params, config: ModelConfig2SegProject):
        Dense = tf.keras.layers.Dense
        encoder1 = BertModelLayer.from_params(bert_params, name="encoder1/bert")
        projector1 = Dense(config.project_dim, activation='relu', name="encoder1/project")

        encoder2 = BertModelLayerWMask.from_params(bert_params, name="encoder2/bert")
        projector2 = Dense(config.project_dim, activation='relu', name="encoder2/project")

        max_seq_len1 = config.max_seq_length1
        max_seq_len2 = config.max_seq_length2
        num_classes = config.num_classes

        l_input_ids1 = keras.layers.Input(shape=(max_seq_len1,), dtype='int32', name="input_ids1")
        l_token_type_ids1 = keras.layers.Input(shape=(max_seq_len1,), dtype='int32', name="segment_ids1")

        l_input_ids2 = keras.layers.Input(shape=(max_seq_len2,), dtype='int32', name="input_ids2")
        l_token_type_ids2 = keras.layers.Input(shape=(max_seq_len2,), dtype='int32', name="segment_ids2")
        dummy_token_type_ids1 = tf.zeros_like(l_token_type_ids1, tf.int32)
        dummy_token_type_ids2 = tf.zeros_like(l_token_type_ids2, tf.int32)
        chunk_start_mask2 = l_token_type_ids2
        input_mask2 = tf.cast(tf.not_equal(l_input_ids2, 0), tf.int32)

        seq_out1 = encoder1([l_input_ids1, dummy_token_type_ids1])
        seq_out1_p = projector1(seq_out1)
        rep1 = tf.reduce_mean(seq_out1_p, axis=1)
        rep1_ = tf.expand_dims(rep1, 1)
        rep1_stacked = tf.tile(rep1_, [1, max_seq_len2, 1])
        attention_mask = ChunkAttentionMaskLayer()(chunk_start_mask2)
        seq_out2 = encoder2([l_input_ids2, dummy_token_type_ids2], attention_mask=attention_mask)
        self.attention_mask = attention_mask

        rep2_stacked = projector2(seq_out2)
        vtf = VectorThreeFeature()
        feature_rep = vtf((rep1_stacked, rep2_stacked))

        local_decisions = TwoLayerDense(bert_params.hidden_size, num_classes,
                                        name="local_decision")(feature_rep)
        B, _ = get_shape_list2(l_input_ids1)
        is_valid_decision = chunk_start_mask2 * input_mask2
        self.is_valid_decision = is_valid_decision
        self.local_decisions = local_decisions
        combine_local_decisions = self.combine_local_decisions_layer()
        self.cld = combine_local_decisions
        output = combine_local_decisions([local_decisions, is_valid_decision])
        inputs = (l_input_ids1, l_token_type_ids1, l_input_ids2, l_token_type_ids2)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        self.model: keras.Model = model
        self.l_bert_list = [encoder1, encoder2]

    def get_keras_model(self):
        return self.model

    def init_checkpoint(self, init_checkpoint):
        l_bert1, l_bert2 = self.l_bert_list
        load_stock_weights(l_bert1, init_checkpoint, n_expected_restore=197)
        load_stock_weights_for_wmask(l_bert2, init_checkpoint, n_expected_restore=197)

    def get_debug_vars(self):
        return {'attention_mask': self.attention_mask}