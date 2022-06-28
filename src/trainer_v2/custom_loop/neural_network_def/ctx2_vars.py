import tensorflow as tf
from tensorflow import keras

from models.transformer.bert_common_v2 import get_shape_list2
from trainer_v2.bert_for_tf2 import BertModelLayer
from trainer_v2.bert_for_tf2.w_mask.transformer import TransformerEncoderLayerWMask
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.modeling_common.bert_common import define_bert_input, load_stock_weights, \
    load_stock_weights_encoder_only
from trainer_v2.custom_loop.modeling_common.network_utils import ChunkAttentionMaskLayerFreeP, TwoLayerDense
from trainer_v2.custom_loop.neural_network_def.inner_network import ClassificationModelIF

KerasDense = tf.keras.layers.Dense



class CtxChunkInteraction2(ClassificationModelIF):
    # Encoder2's output is directly compared
    def __init__(self, decision_combine_layer):
        super(CtxChunkInteraction2, self).__init__()
        self.decision_combine_layer = decision_combine_layer

    def build_model(self, bert_params, model_config):
        config = model_config
        encoder1_lower = BertModelLayer.from_params(bert_params, name="encoder1_lower/bert")
        encoder2_lower = BertModelLayer.from_params(bert_params, name="encoder2_lower/bert")
        encoder2_lower.trainable = False
        encoder_upper = TransformerEncoderLayerWMask.from_params(
            bert_params,
            name="encoder_upper/encoder"
        )

        num_classes = config.num_classes
        l_input_ids1, l_token_type_ids1 = define_bert_input(config.max_seq_length1, "1")

        l_input_ids2, l_token_type_ids2 = define_bert_input(config.max_seq_length2, "2")
        input_mask1 = tf.cast(tf.not_equal(l_input_ids1, 0), tf.int32)
        input_mask2 = tf.cast(tf.not_equal(l_input_ids2, 0), tf.int32)
        input_mask_concat = tf.concat([input_mask1, input_mask2], axis=1)

        # We use token_type_ids to indicate chunks.
        chunk_start_mask1 = l_token_type_ids1
        chunk_start_mask2 = l_token_type_ids2
        # value indicates new chunk (disconnect mask)
        # More 0 indicates wide attentions

        dummy_token_type_ids1 = tf.zeros_like(l_token_type_ids1, tf.int32)
        dummy_token_type_ids2 = tf.zeros_like(l_token_type_ids2, tf.int32)

        rep1 = encoder1_lower([l_input_ids1, dummy_token_type_ids1])
        rep1 = tf.zeros_like(rep1)
        rep2 = encoder2_lower([l_input_ids2, dummy_token_type_ids2])
        concat_rep = tf.concat([rep1, rep2], axis=1)
        is_new_chunk_start = tf.concat([chunk_start_mask1, chunk_start_mask2], axis=1)
        is_h = tf.concat([tf.zeros_like(l_token_type_ids1, tf.int32),
                          tf.ones_like(l_token_type_ids2, tf.int32),
                          ], axis=1)
        is_new_h_chunk_start = is_h * is_new_chunk_start
        attention_mask = ChunkAttentionMaskLayerFreeP()([chunk_start_mask1, chunk_start_mask2])
        attention_mask = tf.transpose(attention_mask, [0, 2, 1])
        self.attention_mask = attention_mask
        decision_rep = encoder_upper(concat_rep, attention_mask=attention_mask)
        local_decisions = KerasDense(num_classes, name="local_decision", activation=tf.nn.softmax)(decision_rep)

        # [B, S, num_classes]

        is_valid_decision = is_new_h_chunk_start * input_mask_concat
        self.is_valid_decision = is_valid_decision
        self.local_decisions = local_decisions
        output = self.decision_combine_layer()(local_decisions, is_valid_decision)
        inputs = (l_input_ids1, l_token_type_ids1, l_input_ids2, l_token_type_ids2)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        self.inputs = inputs
        self.model: keras.Model = model
        self.l_bert_list = [encoder1_lower, encoder2_lower]
        self.l_transformer = encoder_upper

    def get_keras_model(self):
        return self.model

    def init_checkpoint(self, init_checkpoint):
        for l_bert in self.l_bert_list:
            load_stock_weights(l_bert, init_checkpoint, n_expected_restore=197)
        n_embedding_vars = 5
        load_stock_weights_encoder_only(self.l_transformer, init_checkpoint, n_expected_restore=197 - n_embedding_vars)

    def get_debug_vars(self):
        d = {
            'attention_mask': self.attention_mask,
            'is_valid_decision': self.is_valid_decision
        }
        # for i in range(12):
        #     enc_layer: SingleTransformerEncoderLayerWMask = self.l_transformer.encoder_layers[i]
        #     attn_layer: AttentionLayerWMask = enc_layer.self_attention_layer.attention_layer
        #
        #     d['layer_{}_attn_scores'.format(i)] = attn_layer.attention_scores
        #     d['layer_{}_attn_probs'.format(i)] = attn_layer.attention_probs
        return d



class CtxChunkInteraction3(ClassificationModelIF):
    # Encoder2's output is directly compared
    def __init__(self, decision_combine_layer):
        super(CtxChunkInteraction3, self).__init__()
        self.decision_combine_layer = decision_combine_layer

    def build_model(self, bert_params, model_config):
        config = model_config
        encoder1_lower = BertModelLayer.from_params(bert_params, name="encoder1_lower/bert")
        encoder2_lower = BertModelLayer.from_params(bert_params, name="encoder2_lower/bert")
        encoder2_lower.trainable = False
        encoder_upper = TransformerEncoderLayerWMask.from_params(
            bert_params,
            name="encoder_upper/encoder"
        )

        num_classes = config.num_classes
        l_input_ids1, l_token_type_ids1 = define_bert_input(config.max_seq_length1, "1")
        l_input_ids2, l_token_type_ids2 = define_bert_input(config.max_seq_length2, "2")
        input_mask1 = tf.cast(tf.not_equal(l_input_ids1, 0), tf.int32)
        input_mask2 = tf.cast(tf.not_equal(l_input_ids2, 0), tf.int32)
        input_mask_concat = tf.concat([input_mask1, input_mask2], axis=1)

        # We use token_type_ids to indicate chunks.
        chunk_start_mask1 = l_token_type_ids1
        chunk_start_mask2 = l_token_type_ids2
        one_at_begining = tf.concat(
            [tf.ones([1, 1], tf.int32), tf.zeros([1, config.max_seq_length2-1], tf.int32)], axis=1)
        chunk_start_mask2 = chunk_start_mask2 * one_at_begining

        # value indicates new chunk (disconnect mask)
        # More 0 indicates wide attentions

        dummy_token_type_ids1 = tf.zeros_like(l_token_type_ids1, tf.int32)
        dummy_token_type_ids2 = tf.zeros_like(l_token_type_ids2, tf.int32)

        rep1 = encoder1_lower([l_input_ids1, dummy_token_type_ids1])
        rep2 = encoder2_lower([l_input_ids2, dummy_token_type_ids2])
        concat_rep = tf.concat([rep1, rep2], axis=1)
        is_new_chunk_start = tf.concat([chunk_start_mask1, chunk_start_mask2], axis=1)
        is_h = tf.concat([tf.zeros_like(l_token_type_ids1, tf.int32),
                          tf.ones_like(l_token_type_ids2, tf.int32),
                          ], axis=1)
        is_new_h_chunk_start = is_h * is_new_chunk_start
        attention_mask = ChunkAttentionMaskLayerFreeP()([chunk_start_mask1, chunk_start_mask2])
        self.attention_mask = attention_mask
        c_log.info("Use original attention_mask")
        decision_rep = encoder_upper(concat_rep, attention_mask=attention_mask)
        local_decisions = KerasDense(num_classes, name="local_decision", activation=tf.nn.softmax)(decision_rep)

        # [B, S, num_classes]

        is_valid_decision = is_new_h_chunk_start * input_mask_concat
        self.is_valid_decision = is_valid_decision
        self.local_decisions = local_decisions
        output = self.decision_combine_layer()(local_decisions, is_valid_decision)
        inputs = (l_input_ids1, l_token_type_ids1, l_input_ids2, l_token_type_ids2)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        self.inputs = inputs
        self.model: keras.Model = model
        self.l_bert_list = [encoder1_lower, encoder2_lower]
        self.l_transformer = encoder_upper

    def get_keras_model(self):
        return self.model

    def init_checkpoint(self, init_checkpoint):
        for l_bert in self.l_bert_list:
            load_stock_weights(l_bert, init_checkpoint, n_expected_restore=197)
        n_embedding_vars = 5
        load_stock_weights_encoder_only(self.l_transformer, init_checkpoint, n_expected_restore=197 - n_embedding_vars)


class CtxChunkInteraction4(ClassificationModelIF):
    # Encoder2's output is directly compared
    def __init__(self, decision_combine_layer):
        super(CtxChunkInteraction4, self).__init__()
        self.decision_combine_layer = decision_combine_layer

    def build_model(self, bert_params, model_config):
        config = model_config
        encoder1_lower = BertModelLayer.from_params(bert_params, name="encoder1_lower/bert")
        encoder2_lower = BertModelLayer.from_params(bert_params, name="encoder2_lower/bert")
        encoder2_lower.trainable = False
        num_classes = config.num_classes
        l_input_ids1, l_token_type_ids1 = define_bert_input(config.max_seq_length1, "1")

        l_input_ids2, l_token_type_ids2 = define_bert_input(config.max_seq_length2, "2")
        input_mask1 = tf.cast(tf.not_equal(l_input_ids1, 0), tf.int32)
        input_mask2 = tf.cast(tf.not_equal(l_input_ids2, 0), tf.int32)
        input_mask_concat = tf.concat([input_mask1, input_mask2], axis=1)

        # We use token_type_ids to indicate chunks.
        chunk_start_mask1 = l_token_type_ids1
        chunk_start_mask2 = l_token_type_ids2
        # value indicates new chunk (disconnect mask)
        # More 0 indicates wide attentions

        dummy_token_type_ids1 = tf.zeros_like(l_token_type_ids1, tf.int32)
        dummy_token_type_ids2 = tf.zeros_like(l_token_type_ids2, tf.int32)

        rep1 = encoder1_lower([l_input_ids1, dummy_token_type_ids1])
        rep1 = tf.zeros_like(rep1)
        rep2 = encoder2_lower([l_input_ids2, dummy_token_type_ids2])
        concat_rep = tf.concat([rep1, rep2], axis=1)
        is_new_chunk_start = tf.concat([chunk_start_mask1, chunk_start_mask2], axis=1)
        is_h = tf.concat([tf.zeros_like(l_token_type_ids1, tf.int32),
                          tf.ones_like(l_token_type_ids2, tf.int32),
                          ], axis=1)
        is_new_h_chunk_start = is_h * is_new_chunk_start
        attention_mask = ChunkAttentionMaskLayerFreeP()([chunk_start_mask1, chunk_start_mask2])
        attention_mask = tf.transpose(attention_mask, [0, 2, 1])
        self.attention_mask = attention_mask
        local_decisions = TwoLayerDense(bert_params.hidden_size, num_classes,
                                        name="local_decision")(concat_rep)

        # [B, S, num_classes]

        is_valid_decision = is_new_h_chunk_start * input_mask_concat
        self.is_valid_decision = is_valid_decision
        self.local_decisions = local_decisions
        output = self.decision_combine_layer()(local_decisions, is_valid_decision)
        inputs = (l_input_ids1, l_token_type_ids1, l_input_ids2, l_token_type_ids2)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        self.inputs = inputs
        self.model: keras.Model = model
        self.l_bert_list = [encoder1_lower, encoder2_lower]

    def get_keras_model(self):
        return self.model

    def init_checkpoint(self, init_checkpoint):
        for l_bert in self.l_bert_list:
            load_stock_weights(l_bert, init_checkpoint, n_expected_restore=197)



class CtxChunkInteraction5(ClassificationModelIF):
    # Encoder2's output is directly compared
    def __init__(self, decision_combine_layer):
        super(CtxChunkInteraction5, self).__init__()
        self.decision_combine_layer = decision_combine_layer

    def build_model(self, bert_params, model_config):
        config = model_config
        encoder1_lower = BertModelLayer.from_params(bert_params, name="encoder1_lower/bert")
        encoder2_lower = BertModelLayer.from_params(bert_params, name="encoder2_lower/bert")
        encoder2_lower.trainable = False
        num_classes = config.num_classes
        l_input_ids1, l_token_type_ids1 = define_bert_input(config.max_seq_length1, "1")
        l_input_ids2, l_token_type_ids2 = define_bert_input(config.max_seq_length2, "2")

        input_mask1 = tf.cast(tf.not_equal(l_input_ids1, 0), tf.int32)
        input_mask2 = tf.cast(tf.not_equal(l_input_ids2, 0), tf.int32)
        input_mask_concat = tf.concat([input_mask1, input_mask2], axis=1)

        # We use token_type_ids to indicate chunks.
        chunk_start_mask1 = l_token_type_ids1
        chunk_start_mask2 = l_token_type_ids2
        # value indicates new chunk (disconnect mask)
        # More 0 indicates wide attentions

        dummy_token_type_ids1 = tf.zeros_like(l_token_type_ids1, tf.int32)
        dummy_token_type_ids2 = tf.zeros_like(l_token_type_ids2, tf.int32)

        rep1 = encoder1_lower([l_input_ids1, dummy_token_type_ids1])
        rep1 = tf.zeros_like(rep1)
        rep2 = encoder2_lower([l_input_ids2, dummy_token_type_ids2])
        concat_rep = tf.concat([rep1, rep2], axis=1)
        is_new_chunk_start = tf.concat([chunk_start_mask1, chunk_start_mask2], axis=1)
        is_h = tf.concat([tf.zeros_like(l_token_type_ids1, tf.int32),
                          tf.ones_like(l_token_type_ids2, tf.int32),
                          ], axis=1)
        is_new_h_chunk_start = is_h * is_new_chunk_start
        attention_mask = ChunkAttentionMaskFreePH()([chunk_start_mask1, chunk_start_mask2])
        self.attention_mask = attention_mask
        local_decisions = TwoLayerDense(bert_params.hidden_size, num_classes,
                                        name="local_decision")(concat_rep)

        # [B, S, num_classes]

        is_valid_decision = is_new_h_chunk_start * input_mask_concat
        self.is_valid_decision = is_valid_decision
        self.local_decisions = local_decisions
        output = self.decision_combine_layer()(local_decisions, is_valid_decision)
        inputs = (l_input_ids1, l_token_type_ids1, l_input_ids2, l_token_type_ids2)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        self.inputs = inputs
        self.model: keras.Model = model
        self.l_bert_list = [encoder1_lower, encoder2_lower]

    def get_keras_model(self):
        return self.model

    def init_checkpoint(self, init_checkpoint):
        for l_bert in self.l_bert_list:
            load_stock_weights(l_bert, init_checkpoint, n_expected_restore=197)



class ChunkAttentionMaskFreePH(tf.keras.layers.Layer):
    def call(self, inputs, *args, **kwargs):
        p_array, h_array = inputs
        B, L1 = get_shape_list2(p_array)
        B, L2 = get_shape_list2(h_array)
        mask00 = tf.ones([B, L1, L1], tf.int32)
        mask01 = tf.zeros([B, L1, L2], tf.int32)
        mask10 = tf.ones([B, L2, L1], tf.int32)
        mask11 = tf.ones([B, L2, L2], tf.int32)
        output_mak = tf.concat([
            tf.concat([mask00, mask01], axis=2),
            tf.concat([mask10, mask11], axis=2)
        ], axis=1)
        return output_mak



class CtxChunkInteraction6(ClassificationModelIF):
    # Encoder2's output is directly compared
    def __init__(self, decision_combine_layer):
        super(CtxChunkInteraction6, self).__init__()
        self.decision_combine_layer = decision_combine_layer

    def build_model(self, bert_params, model_config):
        config = model_config
        encoder1_lower = BertModelLayer.from_params(bert_params, name="encoder1_lower/bert")
        encoder2_lower = BertModelLayer.from_params(bert_params, name="encoder2_lower/bert")
        encoder2_lower.trainable = False
        num_classes = config.num_classes
        l_input_ids1, l_token_type_ids1 = define_bert_input(config.max_seq_length1, "1")

        l_input_ids2, l_token_type_ids2 = define_bert_input(config.max_seq_length2, "2")
        input_mask1 = tf.cast(tf.not_equal(l_input_ids1, 0), tf.int32)
        input_mask2 = tf.cast(tf.not_equal(l_input_ids2, 0), tf.int32)
        input_mask_concat = tf.concat([input_mask1, input_mask2], axis=1)

        # We use token_type_ids to indicate chunks.
        chunk_start_mask1 = l_token_type_ids1
        chunk_start_mask2 = l_token_type_ids2
        # value indicates new chunk (disconnect mask)
        # More 0 indicates wide attentions

        dummy_token_type_ids1 = tf.zeros_like(l_token_type_ids1, tf.int32)
        dummy_token_type_ids2 = tf.zeros_like(l_token_type_ids2, tf.int32)

        rep1 = encoder1_lower([l_input_ids1, dummy_token_type_ids1])
        rep2 = encoder2_lower([l_input_ids2, dummy_token_type_ids2])
        concat_rep = tf.concat([rep1, rep2], axis=1)
        is_new_chunk_start = tf.concat([chunk_start_mask1, chunk_start_mask2], axis=1)
        is_h = tf.concat([tf.zeros_like(l_token_type_ids1, tf.int32),
                          tf.ones_like(l_token_type_ids2, tf.int32),
                          ], axis=1)
        is_new_h_chunk_start = is_h * is_new_chunk_start
        attention_mask = ChunkAttentionMaskFreePH()([chunk_start_mask1, chunk_start_mask2])
        self.attention_mask = attention_mask
        local_decisions = TwoLayerDense(bert_params.hidden_size, num_classes,
                                        name="local_decision")(concat_rep)

        # [B, S, num_classes]

        is_valid_decision = is_new_h_chunk_start * input_mask_concat
        self.is_valid_decision = is_valid_decision
        self.local_decisions = local_decisions
        output = self.decision_combine_layer()(local_decisions, is_valid_decision)
        inputs = (l_input_ids1, l_token_type_ids1, l_input_ids2, l_token_type_ids2)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        self.inputs = inputs
        self.model: keras.Model = model
        self.l_bert_list = [encoder1_lower, encoder2_lower]

    def get_keras_model(self):
        return self.model

    def init_checkpoint(self, init_checkpoint):
        for l_bert in self.l_bert_list:
            load_stock_weights(l_bert, init_checkpoint, n_expected_restore=197)
