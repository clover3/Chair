
import tensorflow as tf
from bert.embeddings import BertEmbeddingsLayer
from tensorflow import keras

from models.transformer.bert_common_v2 import get_shape_list2
from trainer_v2.bert_for_tf2.w_mask.model_bert_mask import load_stock_weights_nc
from trainer_v2.custom_loop.modeling_common.network_utils import VectorThreeFeature, TwoLayerDense
from trainer_v2.custom_loop.neural_network_def.inner_network import ClassificationModelIF
from trainer_v2.custom_loop.neural_network_def.no_attn import BertModelLayerNoAttn
from trainer_v2.custom_loop.neural_network_def.siamese import ModelConfig2SegProject


def expand_tile(t, dim, n_dim, repeat_n):
    t2 = tf.expand_dims(t, dim)
    tile_map = [1 for _ in range(n_dim+1)]
    tile_map[dim] = repeat_n
    return tf.tile(t2, tile_map)


# Projection + Mean+ Concat
class SingleToken2D(ClassificationModelIF):
    def __init__(self, combine_local_decisions_layer):
        super(SingleToken2D, self).__init__()
        self.combine_local_decisions_layer = combine_local_decisions_layer

    def build_model(self, bert_params, config: ModelConfig2SegProject):
        Dense = tf.keras.layers.Dense
        encoder1 = BertModelLayerNoAttn.from_params(bert_params, name="encoder1/bert")
        encoder2 = BertModelLayerNoAttn.from_params(bert_params, name="encoder2/bert")

        max_seq_len1 = config.max_seq_length1
        max_seq_len2 = config.max_seq_length2
        num_classes = config.num_classes

        l_input_ids1 = keras.layers.Input(shape=(max_seq_len1,), dtype='int32', name="input_ids1")
        l_token_type_ids1 = keras.layers.Input(shape=(max_seq_len1,), dtype='int32', name="segment_ids1")
        input_mask1 = tf.cast(tf.not_equal(l_input_ids1, 0), tf.int32)

        l_input_ids2 = keras.layers.Input(shape=(max_seq_len2,), dtype='int32', name="input_ids2")
        l_token_type_ids2 = keras.layers.Input(shape=(max_seq_len2,), dtype='int32', name="segment_ids2")
        input_mask2 = tf.cast(tf.not_equal(l_input_ids2, 0), tf.int32)

        B, _ = get_shape_list2(l_input_ids1)
        rep1_stacked = encoder1([l_input_ids1, l_token_type_ids1])
        rep1_4d = expand_tile(rep1_stacked, 2, 3, max_seq_len2)  # [B, L1, L2, H]

        rep2_stacked = encoder2([l_input_ids2, l_token_type_ids2])
        rep2_4d = expand_tile(rep2_stacked, 1, 3, max_seq_len1)  # [B, L1, L2, H]
        vtf = VectorThreeFeature()
        feature_rep = vtf((rep1_4d, rep2_4d))

        local_decisions = TwoLayerDense(bert_params.hidden_size, num_classes,
                                        name="local_decision")(feature_rep) # [B, L1, L2, C]

        ld = tf.reduce_max(local_decisions, axis=1)

        self.local_decisions = local_decisions
        combine_ld_input2_wise = self.combine_local_decisions_layer()
        self.cld = combine_ld_input2_wise
        output = combine_ld_input2_wise([ld, input_mask2])
        inputs = (l_input_ids1, l_token_type_ids1, l_input_ids2, l_token_type_ids2)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        self.model: keras.Model = model
        self.l_bert_list = [encoder1, encoder2]

    def get_keras_model(self):
        return self.model

    def init_checkpoint(self, init_checkpoint):
        l_bert1, l_bert2 = self.l_bert_list
        load_stock_weights_nc(l_bert1, init_checkpoint, n_expected_restore=77)
        load_stock_weights_nc(l_bert2, init_checkpoint, n_expected_restore=77)





# Projection + Mean+ Concat
class SingleToken2DEmbOnly(ClassificationModelIF):
    def __init__(self, combine_local_decisions_layer):
        super(SingleToken2DEmbOnly, self).__init__()
        self.combine_local_decisions_layer = combine_local_decisions_layer

    def build_model(self, bert_params, config: ModelConfig2SegProject):
        encoder1 = BertEmbeddingsLayer.from_params(
            bert_params,
            name="encoder1/bert/embeddings"
        )
        encoder2 = BertEmbeddingsLayer.from_params(
            bert_params,
            name="encoder2/bert/embeddings"
        )

        max_seq_len1 = config.max_seq_length1
        max_seq_len2 = config.max_seq_length2
        num_classes = config.num_classes

        l_input_ids1 = keras.layers.Input(shape=(max_seq_len1,), dtype='int32', name="input_ids1")
        l_token_type_ids1 = keras.layers.Input(shape=(max_seq_len1,), dtype='int32', name="segment_ids1")
        input_mask1 = tf.cast(tf.not_equal(l_input_ids1, 0), tf.int32)

        l_input_ids2 = keras.layers.Input(shape=(max_seq_len2,), dtype='int32', name="input_ids2")
        l_token_type_ids2 = keras.layers.Input(shape=(max_seq_len2,), dtype='int32', name="segment_ids2")
        input_mask2 = tf.cast(tf.not_equal(l_input_ids2, 0), tf.int32)

        rep1_stacked = encoder1([l_input_ids1, l_token_type_ids1], mask=input_mask1)
        rep2_stacked = encoder2([l_input_ids2, l_token_type_ids2], mask=input_mask2)

        rep1_4d = expand_tile(rep1_stacked, 2, 3, max_seq_len2)  # [B, L1, L2, H]
        rep2_4d = expand_tile(rep2_stacked, 1, 3, max_seq_len1)  # [B, L1, L2, H]
        vtf = VectorThreeFeature()
        feature_rep = vtf((rep1_4d, rep2_4d))
        local_decisions = TwoLayerDense(bert_params.hidden_size, num_classes,
                                        name="local_decision")(feature_rep)  # [B, L1, L2, C]

        ld = tf.reduce_max(local_decisions, axis=1)
        B, _ = get_shape_list2(l_input_ids1)
        self.local_decisions = local_decisions
        combine_ld_input2_wise = self.combine_local_decisions_layer()
        self.cld = combine_ld_input2_wise
        output = combine_ld_input2_wise([ld, input_mask2])
        inputs = (l_input_ids1, l_token_type_ids1, l_input_ids2, l_token_type_ids2)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        self.model: keras.Model = model
        self.l_bert_list = [encoder1, encoder2]

    def get_keras_model(self):
        return self.model

    def init_checkpoint(self, init_checkpoint):
        l_bert1, l_bert2 = self.l_bert_list
        load_stock_weights_nc(l_bert1, init_checkpoint, n_expected_restore=5)
        load_stock_weights_nc(l_bert2, init_checkpoint, n_expected_restore=5)


