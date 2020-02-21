import tensorflow as tf

import models.transformer.bert_common_v2 as bc
from tlm.model import base
from tlm.model.base import mimic_pooling
from tlm.model.units import Embedding, ForwardLayer


# Option 1 : Only at the first layer vs Every layer
# Option 2 :




class TopicLayerAllLayer:
    def __init__(self, initializer, n_topics, hidden_size, topic_emb_len, n_layers, use_one_hot_embeddings):
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.initializer = initializer
        self.topic_embedding_size = hidden_size * topic_emb_len * n_layers
        self.topic_emb_len = topic_emb_len
        self.hidden_size = hidden_size
        self.n_topics = n_topics
        self.n_layers = n_layers

        self.topic_embedding = tf.Variable(self.initializer(shape=(self.n_topics, self.topic_embedding_size),
                                                            dtype=tf.float32),
                                           name="topic_embedding")

    def extend_input_mask(self, input_mask):
        input_shape = bc.get_shape_list2(input_mask)
        batch_size, seq_length = input_shape
        input_mask = tf.concat([input_mask, tf.ones([batch_size, self.topic_emb_len], tf.int32)], axis=1)
        return input_mask

    def add_topic_vector(self, input_tensor, topic_ids):

        return input_tensor

    def apply_topic_vector(self, input_tensor, topic_ids, layer_idx):
        if layer_idx == 0:
            return self.add_topic_vector(input_tensor, topic_ids)
        else:
            input_tensor = bc.reshape_from_matrix2(input_tensor, self.input_shape)
            input_tensor = input_tensor[:, -self.topic_emb_len, :]
            input_tensor = tf.concat([input_tensor, self.topic_tensor[layer_idx]], axis=1)
            input_tensor = bc.reshape_to_matrix(input_tensor)
            return input_tensor

    def get_sequence_output(self, final_output):
        return final_output[:, :-self.topic_emb_len]


class TopicVectorBert(base.BertModelInterface):
    def __init__(self, config, n_layers, use_one_hot_embeddings):
        super(TopicVectorBert, self).__init__()
        self.n_layers = n_layers
        self.all_layer_outputs = []
        self.last_key_layer = None
        self.config = config
        self.embedding = None
        self.layer_list = []
        self.initializer = base.create_initializer(config.initializer_range)
        self.attention_mask = None
        self.use_one_hot_embeddings = use_one_hot_embeddings
        for layer_idx in range(self.n_layers):
            layer = ForwardLayer(self.config, self.initializer)
            self.layer_list.append(layer)

        self.n_topics = config.n_topics
        self.use_topic_all_layer = config.use_topic_all_layer
        self.hidden_size = config.hidden_size
        topic_emb_len = 4

        self.topic_embedding_size = self.hidden_size * topic_emb_len
        self.n_topics = config.n_topics
        self.topic_emb_len = topic_emb_len
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.topic_embedding = tf.Variable(lambda :self.initializer(shape=(self.n_topics, self.topic_embedding_size),
                                                            dtype=tf.float32),
                                           name="topic_embedding")

    def extend_input_mask(self, input_mask):
        input_shape = bc.get_shape_list2(input_mask)
        batch_size, seq_length  = input_shape
        input_mask = tf.concat([input_mask, tf.ones([batch_size, self.topic_emb_len], tf.int32)], axis=1)
        return input_mask

    @classmethod
    def factory(cls, config, is_training, input_ids, input_mask, token_type_ids, use_one_hot_embeddings, features):
        model = TopicVectorBert(config, config.num_hidden_layers, use_one_hot_embeddings)
        topic_ids = features["topic_ids"]
        with tf.compat.v1.variable_scope("bert"):
            model.call(input_ids, input_mask, token_type_ids, topic_ids)
        return model

    def call(self, input_ids, input_mask, segment_ids, topic_ids):
        with tf.compat.v1.variable_scope("embeddings"):
            self.embedding_layer = Embedding(self.config, self.use_one_hot_embeddings)
            input_tensor = self.embedding_layer.apply(input_ids, segment_ids)
            self.embedding_output = input_tensor

        input_mask = self.extend_input_mask(input_mask)
        topic_tensor, _ = bc.embedding_lookup2(topic_ids,
                                            self.n_topics,
                                            self.topic_embedding,
                                            self.topic_embedding_size,
                                            self.use_one_hot_embeddings)
        self.topic_tensor = tf.reshape(topic_tensor, [-1, self.topic_emb_len, self.hidden_size])

        input_tensor = tf.concat([input_tensor, self.topic_tensor], axis=1)
        input_shape = bc.get_shape_list2(input_tensor)
        batch_size, seq_length, _ = input_shape

        with tf.compat.v1.variable_scope("encoder"):
            self.attention_mask = bc.create_attention_mask_from_input_mask2(
                input_tensor, input_mask)
            prev_output = bc.reshape_to_matrix(input_tensor)
            for layer_idx in range(self.n_layers):
                with tf.compat.v1.variable_scope("layer_%d" % layer_idx):
                    layer = self.layer_list[layer_idx]
                    intermediate_output, prev_output = layer.apply(prev_output, batch_size, seq_length,
                                                             self.attention_mask)
                    final_output = bc.reshape_from_matrix2(prev_output, input_shape)
                    self.all_layer_outputs.append(final_output)

        self.embedding_table = self.embedding_layer.embedding_table
        self.sequence_output = final_output[:, :-self.topic_emb_len]
        self.pooled_output = mimic_pooling(self.sequence_output, self.config.hidden_size, self.config.initializer_range)
        return self.sequence_output
