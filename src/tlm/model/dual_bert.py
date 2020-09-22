import tensorflow as tf

from tlm.model.base import BertModelInterface, BertModel, create_initializer
from tlm.model.dual_model_common import dual_model_prefix1, dual_model_prefix2


class DualBertModel(BertModelInterface):
    def __init__(self,
               config,
               is_training,
               input_ids,
               input_mask=None,
               token_type_ids=None,
               use_one_hot_embeddings=True,
               scope=None):
        super(DualBertModel, self).__init__()


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
                    input_ids=input_ids,
                    input_mask=input_mask,
                    token_type_ids=token_type_ids,
                    use_one_hot_embeddings=use_one_hot_embeddings,
            )

        model_1_first_token = model_1.get_sequence_output()[:, 0, :]
        model_2_first_token = model_2.get_sequence_output()[:, 0, :]

        rep = tf.concat([model_1_first_token, model_2_first_token], axis=1)

        self.sequence_output = tf.concat([model_1.get_sequence_output(), model_2.get_sequence_output()], axis=2)
        dense_layer = tf.keras.layers.Dense(config.hidden_size,
                                              activation=tf.keras.activations.tanh,
                                              kernel_initializer=create_initializer(config.initializer_range))
        pooled_output = dense_layer(rep)
        self.pooled_output = pooled_output

