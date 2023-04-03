from typing import Dict
from typing import List, Iterable, Callable, Dict, Tuple, Set

import tensorflow as tf
from bert import Layer
from tensorflow import keras
from trainer_v2.bert_for_tf2 import BertModelLayer
from trainer_v2.per_project.transparency.splade_regression.modeling.regression_modeling import ReluSigmoidMaxReduce
import numpy as np

class BertSparseEncoder:
    def __init__(self, bert_params, dataset_info: Dict):
        l_bert = BertModelLayer.from_params(bert_params, name="bert")
        max_seq_len = dataset_info['max_seq_length']
        num_classes = dataset_info['vocab_size']

        l_input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
        attention_mask = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="attention_mask")
        l_token_type_ids = tf.zeros_like(l_input_ids)
        seq_out = l_bert([l_input_ids, l_token_type_ids])  # [batch_size, max_seq_len, hidden_size]
        self.seq_out = seq_out
        logits = tf.keras.layers.Dense(num_classes, name="bert/project")(seq_out)
        activation_layer = ReluSigmoidMaxReduce()
        output = activation_layer(logits, attention_mask)

        model = keras.Model(inputs=(l_input_ids, attention_mask), outputs=output, name="bert_model")
        self.model: keras.Model = model
        self.l_bert = l_bert


class DummySparseEncoder:
    def __init__(self, dataset_info: Dict):
        max_seq_len = dataset_info['max_seq_length']
        num_classes = dataset_info['vocab_size']

        l_input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
        attention_mask = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="attention_mask")
        l_token_type_ids = tf.zeros_like(l_input_ids)

        batch_size, _ = tf.shape(l_input_ids)
        output = tf.zeros([batch_size, num_classes])
        model = keras.Model(inputs=(l_input_ids, attention_mask), outputs=output, name="dummy_encoder")
        self.model: keras.Model = model


class MaskedLM(tf.keras.layers.Layer):
    def __init__(self,
                 embedding_table,
                 activation=None,
                 initializer='glorot_uniform',
                 output='logits',
                 name=None,
                 **kwargs):
        super(MaskedLM, self).__init__(name=name, **kwargs)
        self.embedding_table = embedding_table
        self.activation = activation
        self.initializer = tf.keras.initializers.get(initializer)
        self._output_type = output

    def build(self, input_shape):
        _vocab_size: object
        self._vocab_size, hidden_size = self.embedding_table.shape
        self.dense = tf.keras.layers.Dense(
            hidden_size,
            activation=self.activation,
            kernel_initializer=self.initializer,
            name='transform/dense')
        self.layer_norm = tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-12, name='transform/LayerNorm')
        # self.project = tf.keras.layers.Dense(
        #     self._vocab_size,
        #     activation=self.activation,
        #     kernel_initializer=self.initializer,
        #     name='transform/project')

        self.output_weights = self.add_weight(
            'output_weights',
            shape=(self._vocab_size, hidden_size),
            initializer='zeros',
            trainable=True)

        self.bias = self.add_weight(
            'output_bias',
            shape=(self._vocab_size,),
            initializer='zeros',
            trainable=True)

        super(MaskedLM, self).build(input_shape)

    def call(self, sequence_data):
        lm_data = self.dense(sequence_data)
        lm_data = self.layer_norm(lm_data)
        # logits = self.project(lm_data)
        lm_data = tf.matmul(lm_data, self.output_weights, transpose_b=True)
        logits = tf.nn.bias_add(lm_data, self.bias)
        return logits


# It output LM like scores, but scores are activated by sigmoid instead of softmax.
class BertLMLogits:
    def __init__(self, bert_params, dataset_info: Dict):
        l_bert = BertModelLayer.from_params(bert_params, name="bert")
        max_seq_len = dataset_info['max_seq_length']
        num_classes = dataset_info['vocab_size']
        l_input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
        attention_mask = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="attention_mask")
        l_token_type_ids = tf.zeros_like(l_input_ids)
        seq_out = l_bert([l_input_ids, l_token_type_ids])  # [batch_size, max_seq_len, hidden_size]
        self.seq_out = seq_out
        embedding_table = l_bert.embeddings_layer.word_embeddings_layer.embeddings
        self.mask_lm = MaskedLM(
            embedding_table, bert_params.intermediate_activation,
            name='cls/predictions')
        logits = self.mask_lm(seq_out)
        activation_layer = ReluSigmoidMaxReduce()
        logits = activation_layer(logits, attention_mask)
        model = keras.Model(inputs=(l_input_ids, attention_mask), outputs=logits, name="bert_model")
        self.model: keras.Model = model

        self.l_bert = l_bert
        self.lm_params = [
            self.mask_lm.dense.weights,
            self.mask_lm.layer_norm.weights,
            [self.mask_lm.output_weights, self.mask_lm.bias]
        ]


class BertSparseEncoderPostMask:
    def __init__(self, bert_params, dataset_info: Dict, mask_indices: List[int]):
        l_bert = BertModelLayer.from_params(bert_params, name="bert")
        max_seq_len = dataset_info['max_seq_length']
        num_classes = dataset_info['vocab_size']

        l_input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
        attention_mask = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="attention_mask")
        l_token_type_ids = tf.zeros_like(l_input_ids)
        seq_out = l_bert([l_input_ids, l_token_type_ids])  # [batch_size, max_seq_len, hidden_size]
        self.seq_out = seq_out
        logits = tf.keras.layers.Dense(num_classes, name="bert/project")(seq_out)
        activation_layer = ReluSigmoidMaxReduce()
        output = activation_layer(logits, attention_mask)

        mask = np.ones([num_classes], np.float32)
        for i in mask_indices:
            mask[i] = 0

        mask_t = tf.expand_dims(tf.constant(mask), 0)
        output = output * mask_t
        model = keras.Model(inputs=(l_input_ids, attention_mask), outputs=output, name="bert_model")
        self.model: keras.Model = model
        self.l_bert = l_bert
