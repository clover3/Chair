import tensorflow as tf

import models.transformer.bert_common_v2 as bc
from tlm.model import base
from tlm.model.base import mimic_pooling
from tlm.model.units import ForwardLayer, Embedding2


class SharedTransformer(tf.keras.layers.Layer):
    def __init__(self, config, use_one_hot_embeddings, **kwargs):
        kwargs['autocast'] = False
        super(SharedTransformer, self).__init__(kwargs)
        self.all_layer_outputs = []
        self.last_key_layer = None
        self.config = config

        self.initializer = base.create_initializer(config.initializer_range)
        self.attention_mask = None
        self.use_one_hot_embeddings = use_one_hot_embeddings
        with tf.compat.v1.variable_scope("layer"):
            self.layer = ForwardLayer(self.config, self.initializer)

    def call(self, input_ids, input_mask, segment_ids):
        with tf.compat.v1.variable_scope("embeddings"):
            self.embedding_layer = Embedding2()
            input_tensor = self.embedding_layer.apply(input_ids, segment_ids,
                                                      self.config.initializer_range,
                                                      self.config.vocab_size,
                                                      self.config.embedding_size,
                                                      self.config.type_vocab_size,
                                                      self.config.max_position_embeddings,
                                                      self.config.hidden_dropout_prob,
                                                      self.use_one_hot_embeddings
                                                      )
            input_tensor = self.embedding_projection(input_tensor)
            self.embedding_output = input_tensor
            input_shape = bc.get_shape_list2(input_tensor)
            batch_size, seq_length, _ = input_shape
        with tf.compat.v1.variable_scope("encoder"):
            self.attention_mask = bc.create_attention_mask_from_input_mask2(
                input_tensor, input_mask)
            prev_output = bc.reshape_to_matrix(input_tensor)
            with tf.compat.v1.variable_scope("layer"):
                intermediate_output, prev_output = self.layer.apply(prev_output, batch_size, seq_length,
                                                               self.attention_mask)
                final_output = bc.reshape_from_matrix2(prev_output, input_shape)
                self.all_layer_outputs.append(final_output)

            for layer_idx in range(1, self.config.num_hidden_layers):
                with tf.compat.v1.variable_scope("layer", reuse=True):
                    intermediate_output, prev_output = self.layer.apply(prev_output, batch_size, seq_length,
                                                             self.attention_mask)
                    final_output = bc.reshape_from_matrix2(prev_output, input_shape)
                    self.all_layer_outputs.append(final_output)

        return prev_output

    def embedding_projection(self, input_tensor):
        with tf.compat.v1.variable_scope("embedding_projection", reuse=True):
            return bc.dense(self.config.hidden_size, self.initializer)(input_tensor)


class Albert(base.BertModelInterface):
    def __init__(self,
                 config, # This is different from BERT config,
                 is_training,
                 use_one_hot_embeddings=True,
                 ):
        super(Albert, self).__init__()
        self.config = config
        if not is_training:
            config.set_attrib("hidden_dropout_prob", 0.0)
            config.set_attrib("attention_probs_dropout_prob", 0.0)
        self.module = SharedTransformer(config, use_one_hot_embeddings)

    @classmethod
    def factory(cls, config, is_training, input_ids, input_mask, token_type_ids, use_one_hot_embeddings):
        model = Albert(config, is_training, use_one_hot_embeddings)
        model.call(input_ids, input_mask, token_type_ids)
        return model

    def call(self, input_ids, input_mask, segment_ids):
        self.module.call(input_ids, input_mask, segment_ids)
        self.sequence_output = self.module.all_layer_outputs[-1]
        self.all_encoder_layers = self.module.all_layer_outputs
        self.all_encoder_layers = []
        self.embedding_output = self.module.embedding_output
        self.pooled_output = mimic_pooling(self.sequence_output, self.config.hidden_size, self.config.initializer_range)
        self.embedding_table = self.module.embedding_layer.embedding_table
        return self.sequence_output


class BertologyFactory:
    def __init__(self, class_ref):
        self.class_ref = class_ref

    def __call__(self, config, is_training, input_ids, input_mask, token_type_ids, use_one_hot_embeddings):
        model = self.class_ref(config, is_training, use_one_hot_embeddings)
        model.call(input_ids, input_mask, token_type_ids)
        return model