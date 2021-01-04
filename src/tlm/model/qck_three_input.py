import tensorflow as tf

from tf_util.lib.tf_funcs import get_pos_only_weight_param
from tlm.model.base import BertModel
from tlm.model.dual_model_common import dual_model_prefix1, dual_model_prefix2


def get_tower1(config, is_training, use_one_hot_embeddings, features):
    input_ids = tf.concat([features["input_ids0"], features["input_ids1"]], axis=1)
    input_mask = tf.concat([features["input_mask0"], features["input_mask1"]], axis=1)
    segment_ids_0 = tf.zeros_like(features["segment_ids0"], tf.int32)
    segment_ids_1 = tf.ones_like(features["segment_ids0"], tf.int32)
    segment_ids = tf.concat([segment_ids_0, segment_ids_1], axis=1)

    model_1 = BertModel(
        config=config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings,
    )
    return model_1.get_pooled_output()


def get_tower2(config, is_training, use_one_hot_embeddings, features):
    # Implement
    transformer_list = []
    pooled_output_list = []
    for i in range(3):
        input_ids = features["input_ids{}".format(i)]
        input_mask = features["input_mask{}".format(i)]
        token_type_ids = features["segment_ids{}".format(i)]

        with tf.compat.v1.variable_scope("sub_module"):
            transformer = BertModel(
                config=config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=token_type_ids,
                use_one_hot_embeddings=use_one_hot_embeddings,
            )
            pooled = transformer.get_sequence_output()[:, 0, :]
            transformer_list.append(transformer)
            pooled_output_list.append(pooled)

    v_1 = pooled_output_list[1]
    v_2 = pooled_output_list[2]
    v_dot = v_1 * v_2  # [batch * hidden_size]
    return v_dot


class HybridQCK:
    def __init__(self,
               config,
               is_training,
               use_one_hot_embeddings=True,
               features=None,
               scope=None):
        with tf.compat.v1.variable_scope(dual_model_prefix1):
            pooled1 = self.get_tower1(config, is_training, use_one_hot_embeddings, features)

        with tf.compat.v1.variable_scope(dual_model_prefix2):
            pooled2 = get_tower2(config, is_training, use_one_hot_embeddings, features)

        def get_dense(hidden_dim):
            return tf.keras.layers.Dense(hidden_dim)

        n_intermediate_dim = config.n_intermediate_dim
        n_dim = config.n_dim
        x1 = get_dense(n_intermediate_dim)(pooled1)
        x1 = get_dense(n_dim)(x1)
        x2 = get_dense(n_intermediate_dim)(pooled2)
        x2 = get_dense(n_dim)(x2)

        k1 = get_pos_only_weight_param([1, n_dim], "k1")
        k2 = get_pos_only_weight_param([1, n_dim], "k2")
        B = tf.compat.v1.get_variable(
                "bias", [1, n_dim],
                initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02)
            )

        h = x1 * k1 + x2 * k2 + B
        prob = tf.reduce_mean(h, axis=1)
        self.prob = prob

    def get_prob(self):
        return self.prob


class DotCK:
    def __init__(self,
               config,
               is_training,
               use_one_hot_embeddings=True,
               features=None,
               ):
        pooled2 = get_tower2(config, is_training, use_one_hot_embeddings, features)
        prob = tf.reduce_mean(pooled2, axis=1)
        self.prob = prob

    def get_prob(self):
        return self.prob
