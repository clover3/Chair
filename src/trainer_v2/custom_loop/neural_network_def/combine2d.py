
import tensorflow as tf
from bert.embeddings import BertEmbeddingsLayer
from tensorflow import keras

from models.transformer.bert_common_v2 import get_shape_list2
from trainer_v2.bert_for_tf2.w_mask.model_bert_mask import load_stock_weights_nc
from trainer_v2.custom_loop.modeling_common.network_utils import VectorThreeFeature, TwoLayerDense
from trainer_v2.custom_loop.neural_network_def.inner_network import BertBasedModelIF
from trainer_v2.custom_loop.neural_network_def.lattice_based import MonoCombiner3d
from trainer_v2.custom_loop.neural_network_def.no_attn import BertModelLayerNoAttn
from trainer_v2.custom_loop.neural_network_def.siamese import ModelConfig2SegProject


def expand_tile(t, dim, n_dim, repeat_n):
    t2 = tf.expand_dims(t, dim)
    tile_map = [1 for _ in range(n_dim+1)]
    tile_map[dim] = repeat_n
    return tf.tile(t2, tile_map)


class ReduceMaxLayer(tf.keras.layers.Layer):
    def call(self, inputs, *args, **kwargs):
        return tf.reduce_max(inputs, axis=1)


class MonoSortCombinerA(tf.keras.layers.Layer):
    def __init__(self):
        super(MonoSortCombinerA, self).__init__()
        self.inner_combiner = MonoCombiner3d(3, 3)

    def call(self, local_decisions, *args, **kwargs):
        max_l_probs = tf.reduce_max(local_decisions, axis=1)  # [batch_size, L2, 3]
        min_l_probs = tf.reduce_min(local_decisions, axis=1)  # [batch_size, L2, 3]
        avg_l_probs = tf.reduce_mean(local_decisions, axis=1)
        rep = tf.concat([max_l_probs, min_l_probs, avg_l_probs], axis=2)  # [batch_size, L2, 9]
        ret = self.inner_combiner(rep)
        return ret


class MonoSortCombinerB(tf.keras.layers.Layer):
    def __init__(self):
        super(MonoSortCombinerB, self).__init__()
        self.inner_combiner = MonoCombiner3d(3, 3)

    def call(self, local_decisions, *args, **kwargs):
        B, L1, L2, D = get_shape_list2(local_decisions)
        t2 = tf.sort(local_decisions, axis=1)
        rep = t2[:, :3]  # [batch_size, 3, L2, 3]
        rep = tf.transpose(rep, [0, 2, 1, 3])
        rep = tf.reshape(rep, [B, L2, 3 * 3])
        return self.inner_combiner(rep)


class SortLattice(tf.keras.layers.Layer):
    def call(self, inputs, *args, **kwargs):
        return tf.reduce_max(inputs, axis=1)


class SingleToken2D(BertBasedModelIF):
    def __init__(self, combine_ld1_layer, combine_ld2_layer):
        super(SingleToken2D, self).__init__()
        self.combine_ld1_layer = combine_ld1_layer
        self.combine_ld2_layer = combine_ld2_layer

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

        ld = self.combine_ld1_layer(local_decisions)
        self.local_decisions = local_decisions
        combine_ld_input2_wise = self.combine_ld2_layer()
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


class SingleToken2DEmbOnly(BertBasedModelIF):
    def __init__(self, combine_ld1_layer_factory, combine_ld2_layer_factory):
        super(SingleToken2DEmbOnly, self).__init__()
        self.combine_ld1_layer_factory = combine_ld1_layer_factory
        self.combine_ld2_layer_factory = combine_ld2_layer_factory

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

        ld = self.combine_ld1_layer_factory()(local_decisions)
        print(ld)

        B, _ = get_shape_list2(l_input_ids1)
        self.local_decisions = local_decisions
        output = self.combine_ld2_layer_factory()([ld, input_mask2])
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


class SingleToken2DNoPE(BertBasedModelIF):
    def __init__(self, combine_ld1_layer, combine_ld2_layer):
        super(SingleToken2DNoPE, self).__init__()
        self.combine_ld1_layer = combine_ld1_layer
        self.combine_ld2_layer = combine_ld2_layer

    def build_model(self, bert_params, config: ModelConfig2SegProject):
        Dense = tf.keras.layers.Dense
        bert_params.use_position_embeddings = False
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

        ld = self.combine_ld1_layer()(local_decisions)
        self.local_decisions = local_decisions
        output = self.combine_ld2_layer()([ld, input_mask2])
        inputs = (l_input_ids1, l_token_type_ids1, l_input_ids2, l_token_type_ids2)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        self.model: keras.Model = model
        self.l_bert_list = [encoder1, encoder2]

    def get_keras_model(self):
        return self.model

    def init_checkpoint(self, init_checkpoint):
        l_bert1, l_bert2 = self.l_bert_list
        load_stock_weights_nc(l_bert1, init_checkpoint, n_expected_restore=76)
        load_stock_weights_nc(l_bert2, init_checkpoint, n_expected_restore=76)


class SingleToken2DWeighting(BertBasedModelIF):
    def __init__(self, combine_ld2_layer_factory):
        super(SingleToken2DWeighting, self).__init__()
        self.combine_ld2_layer_factory = combine_ld2_layer_factory

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

        decision_weights = TwoLayerDense(bert_params.hidden_size, 1, activation1='relu',
                                         activation2=None,
                                         name="weights")(feature_rep)  # [B, L1, L2, 1]

        decision_weights_norm = tf.nn.softmax(decision_weights, axis=1)  # [B, L1, L2, 1]
        weighted_local_decision_flat = local_decisions * decision_weights_norm
        weighted_local_decision = tf.reduce_sum(weighted_local_decision_flat, axis=1)
        B, _ = get_shape_list2(l_input_ids1)
        self.local_decisions = local_decisions
        output = self.combine_ld2_layer_factory()([weighted_local_decision, input_mask2])
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


def get_new_input_ids(l_input_ids, attention_probs, seq_len, m):
    attention_sorted = tf.argsort(attention_probs, axis=2, direction='ASCENDING')
    top_m = attention_sorted[:, :, :m]  # [B, L, 3]
    l_input_ids_ex = tf.tile(tf.expand_dims(l_input_ids, 1), [1, seq_len, 1])
    # l_input_ids1_ex[i, j, k] = l_input_ids1[i, k]
    # [B, L],
    added_input_ids = tf.gather(l_input_ids_ex, top_m, batch_dims=2)  # [B, L, M]
    return tf.concat([tf.expand_dims(l_input_ids, axis=2), added_input_ids], axis=2)


def main():
    l_input_ids = tf.constant([[101, 1002, 1003, 1004, 1005, 102, 0, 0, 0, 0]])
    attention_probs = tf.random.uniform([1, 10, 10])
    seq_len = 10
    new_input_ids = get_new_input_ids(l_input_ids, attention_probs, seq_len)
    print(new_input_ids)


if __name__ == "__main__":
    main()