import tensorflow as tf

from tlm.model.base import BertModelInterface, BertModel
from tlm.model.dual_model_common import dual_model_prefix1, dual_model_prefix2


class TwoInputModelManualCombine(BertModelInterface):
    def __init__(self,
               config,
               is_training,
               input_ids,
               input_mask=None,
               token_type_ids=None,
               use_one_hot_embeddings=True,
               features=None,
               scope=None):
        super(TwoInputModelManualCombine, self).__init__()

        input_ids2 = features["input_ids2"]
        input_mask2 = features["input_mask2"]
        segment_ids2 = features["segment_ids2"]

        with tf.compat.v1.variable_scope(dual_model_prefix1):
            model_1 = BertModel(
                    config=config,
                    is_training=is_training,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    token_type_ids=token_type_ids,
                    use_one_hot_embeddings=use_one_hot_embeddings,
            )

        with tf.compat.v1.variable_scope(dual_model_prefix2):
            model_2 = BertModel(
                    config=config,
                    is_training=is_training,
                    input_ids=input_ids2,
                    input_mask=input_mask2,
                    token_type_ids=segment_ids2,
                    use_one_hot_embeddings=use_one_hot_embeddings,
            )

        model_1_first_token = model_1.get_sequence_output()[:, 0, :]
        model_2_first_token = model_2.get_sequence_output()[:, 0, :]

        def get_dense(hidden_dim):
            return tf.keras.layers.Dense(hidden_dim)

        n_intermediate_dim = config.n_intermediate_dim
        n_dim = config.n_dim

        x1 = get_dense(n_intermediate_dim)(model_1_first_token)
        x1 = get_dense(n_dim)(x1)
        x2 = get_dense(n_intermediate_dim)(model_2_first_token)
        x2 = get_dense(n_dim)(x2)


        def get_pos_only_weight_param(shape, name):
            output_weights = tf.compat.v1.get_variable(
                name, shape,
                initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02)
            )
            return tf.sigmoid(output_weights)

        k1 = get_pos_only_weight_param([1, n_dim], "k1")
        k2 = get_pos_only_weight_param([1, n_dim], "k2")
        B = tf.compat.v1.get_variable(
                "bias", [1, n_dim],
                initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02)
            )
        if config.zero_k2:
            print("Zero k2")
            k2 = 0
        self.score1 = x1
        self.score2 = x2
        h = x1 * k1 + x2 * k2 + B
        if config.no_sigmoid:
            print("Not using sigmoid")
            g = h
        else:
            print("Using sigmoid")
            g = tf.math.sigmoid(h)
        prob = tf.reduce_mean(g, axis=1)
        self.prob = prob

    def get_prob(self):
        return self.prob
