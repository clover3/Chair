

import tensorflow as tf

from tlm.model.base import BertModelInterface, BertModel, create_initializer, get_shape_list
from tlm.model.dual_model_common import dual_model_prefix1, dual_model_prefix2
from tlm.sero.sero_core import SeroEpsilon


class DualSeroBertModel(BertModelInterface):
    def __init__(self,
               sero_config,
               config,
               is_training,
               input_ids,
               input_mask=None,
               token_type_ids=None,
               use_one_hot_embeddings=True,
               scope=None):
        super(DualSeroBertModel, self).__init__()

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
            with tf.compat.v1.variable_scope("sero"):
                model = SeroEpsilon(
                    sero_config,
                    is_training,
                    use_one_hot_embeddings
                )

                batch_size, _ = get_shape_list(input_mask)
                use_context = tf.ones([batch_size, 1], tf.int32)
                input_ids = tf.expand_dims(input_ids, 1)
                input_mask = tf.expand_dims(input_mask, 1)
                segment_ids = tf.expand_dims(token_type_ids, 1)
                sequence_output2 = model.network_stacked(input_ids, input_mask, segment_ids, use_context)

        model_1_first_token = model_1.get_sequence_output()[:, 0, :]
        model_2_first_token = sequence_output2[:, 0, :]

        rep = tf.concat([model_1_first_token, model_2_first_token], axis=1)
        dense_layer = tf.keras.layers.Dense(config.hidden_size,
                                              activation=tf.keras.activations.tanh,
                                              kernel_initializer=create_initializer(config.initializer_range))
        pooled_output = dense_layer(rep)
        self.pooled_output = pooled_output
