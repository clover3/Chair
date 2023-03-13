import tensorflow as tf
from tensorflow import keras

from trainer_v2.bert_for_tf2 import BertModelLayer
from trainer_v2.bert_for_tf2.w_mask.attention import AttentionLayerWMask
from trainer_v2.bert_for_tf2.w_mask.transformer import TransformerEncoderLayerWMask, SingleTransformerEncoderLayerWMask
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.modeling_common.bert_common import define_bert_input, load_stock_weights, \
    load_stock_weights_encoder_only
from trainer_v2.custom_loop.modeling_common.network_utils import MeanProjectionEnc, TileAfterExpandDims, \
    VectorThreeFeature, TwoLayerDense, ChunkAttentionMaskLayerFreeP
from trainer_v2.custom_loop.neural_network_def.inner_network import BertBasedModelIF

KerasDense = tf.keras.layers.Dense


class CtxSingle(BertBasedModelIF):
    # Encoder2's output is directly compared
    def __init__(self, decision_combine_layer):
        super(CtxSingle, self).__init__()
        self.decision_combine_layer = decision_combine_layer

    def build_model(self, bert_params, model_config):
        config = model_config
        encoder1 = MeanProjectionEnc(bert_params, config.project_dim, "encoder1")
        encoder2_lower = BertModelLayer.from_params(bert_params, name="{}/bert".format("encoder2_a"))
        encoder2_lower.trainable = False
        num_classes = config.num_classes
        l_input_ids1, l_token_type_ids1 = define_bert_input(config.max_seq_length1, "1")
        l_input_ids2, l_token_type_ids2 = define_bert_input(config.max_seq_length2, "2")

        rep1 = encoder1([l_input_ids1, l_token_type_ids1])
        # rep2 = encoder2_lower([l_input_ids2, l_token_type_ids2], training=False)
        rep2 = encoder2_lower([l_input_ids2, l_token_type_ids2])
        self.projector: KerasDense = KerasDense(model_config.project_dim, activation='relu', name="encoder2/project")
        rep2_stacked = self.projector(rep2)
        rep1_stacked = TileAfterExpandDims(1, [1, config.max_seq_length2, 1])(rep1)

        vtf = VectorThreeFeature()
        feature_rep = vtf((rep1_stacked, rep2_stacked))

        local_decisions = TwoLayerDense(bert_params.hidden_size, num_classes, name="local_decision")(feature_rep)
        self.local_decisions = local_decisions

        input_mask2 = tf.cast(tf.not_equal(l_input_ids2, 0), tf.float32)
        output = self.decision_combine_layer()(local_decisions, input_mask2)
        inputs = (l_input_ids1, l_token_type_ids1, l_input_ids2, l_token_type_ids2)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        self.inputs = inputs
        self.model: keras.Model = model
        self.l_bert_list = [encoder1.l_bert, encoder2_lower]

    def get_keras_model(self):
        return self.model

    def init_checkpoint(self, init_checkpoint):
        for l_bert in self.l_bert_list:
            load_stock_weights(l_bert, init_checkpoint, n_expected_restore=197)


def add_0s(base_mask, window_size, max_seq_length):
    mask = []
    for i in range(max_seq_length):
        rel_loc = i % window_size
        if rel_loc == 0:
            mask.append(1)
        else:
            mask.append(0)

    mask_ex = tf.reshape(mask, [1, max_seq_length])
    return tf.minimum(base_mask, mask_ex)


class ModelConfigCtxChunk:
    max_seq_length1 = 200
    max_seq_length2 = 100
    num_classes = 3
    window_len = 4


class CtxChunkInteraction(BertBasedModelIF):
    # Encoder2's output is directly compared
    def __init__(self, decision_combine_layer):
        super(CtxChunkInteraction, self).__init__()
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

        try:
            wl = model_config.window_len
            c_log.info("Using additional zero masking with wl={}".format(wl))
            # (wl-1) / wl of array will be zero (in addition to original mask)
            # e.g, if wl==3,  [1, 0, 0] mask will be apply by taking minimum
            chunk_start_mask1 = add_0s(chunk_start_mask1, wl, config.max_seq_length1) # [1, 0, 0, 1, 0, 0]
            chunk_start_mask2 = add_0s(chunk_start_mask2, wl, config.max_seq_length2)  # [1, 0, 0, 1, 0, 0]
        except AttributeError:
            c_log.info("Not using additional zero masking")

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
        for i in range(12):
            enc_layer: SingleTransformerEncoderLayerWMask = self.l_transformer.encoder_layers[i]
            attn_layer: AttentionLayerWMask = enc_layer.self_attention_layer.attention_layer
            d['layer_{}_attn_scores'.format(i)] = attn_layer.attention_scores
            d['layer_{}_attn_probs'.format(i)] = attn_layer.attention_probs
        return d