import copy

import tensorflow as tf

from tlm.model.base import BertModelInterface, BertModel


class ProjectedMaxPooling(BertModelInterface):
    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 use_one_hot_embeddings=True,
                 scope=None):
        super(ProjectedMaxPooling, self).__init__()
        config = copy.deepcopy(config)
        self.config = config
        self.vector_size = config.vector_size

        self.bert_model = BertModel(config, is_training,
                                    input_ids, input_mask, token_type_ids,
                                    use_one_hot_embeddings, scope)

    def get_pooled_output(self):
        seq_output = self.bert_model.get_sequence_output()
        # projected = tf.keras.layers.Dense(self.vector_size,
        #                                   activation=tf.keras.activations.tanh,
        #                                   kernel_initializer=
        #                                   create_initializer(self.config.initializer_range))(seq_output)
        projected = seq_output
        pooled_output = tf.reduce_mean(projected, axis=1)
        return pooled_output

