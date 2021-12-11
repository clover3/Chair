import os
import re

import tensorflow as tf

from cpath import data_path
from models.keras_model.bert_keras.modular_bert_v2 import MyLayer, BertLayer
from models.keras_model.bert_keras.v1_load_util import load_stock_weights
from models.transformer.bert_common_v2 import create_initializer, get_activation
from tlm.model.dual_model_common import dual_model_prefix1, dual_model_prefix2
from tlm.model_cnfig import JsonConfig


class EncoderWithBias(MyLayer):
    def __init__(self, config, use_one_hot_embeddings):
        super(EncoderWithBias, self).__init__()
        with tf.compat.v1.variable_scope("bert"):
            self.bert_layer = BertLayer(config, use_one_hot_embeddings, True)
            with tf.compat.v1.variable_scope("pooler") as name_scope:
                self.pooler = tf.keras.layers.Dense(config.hidden_size,
                                                            activation=tf.keras.activations.tanh,
                                                            kernel_initializer=create_initializer(
                                                                config.initializer_range),
                                                            name=name_scope.name + "/dense"
                                                            )
        self.mlp_layer1 = tf.keras.layers.Dense(config.intermediate_size,
                                                        activation=get_activation(config.hidden_act))
        self.mlp_layer2 = tf.keras.layers.Dense(config.q_voca_size)
        self.bias_dense = tf.keras.layers.Dense(1, activation=tf.keras.activations.tanh,
                                                        kernel_initializer=create_initializer(config.initializer_range))

    def call(self, inputs, **kwargs):
        with tf.compat.v1.variable_scope("bert"):
            sequence_output = self.bert_layer.call(inputs)
            self.sequence_output = sequence_output
            last_layer = sequence_output[-1]
            first_token_tensor = tf.squeeze(last_layer[:, 0:1, :], axis=1)
            pooled = self.pooler(first_token_tensor)
            h = self.mlp_layer1(pooled)
            qtype_vector = self.mlp_layer2(h)
            bias = self.bias_dense(pooled)
            return qtype_vector, bias


class QDE4(MyLayer):
    def __init__(self, config, use_one_hot_embeddings, is_training=False):
        super(QDE4, self).__init__()
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0
        self.hidden_size = config.hidden_size
        with tf.compat.v1.variable_scope(dual_model_prefix1):
            self.q_encoder = EncoderWithBias(config, use_one_hot_embeddings)

        with tf.compat.v1.variable_scope(dual_model_prefix2):
            self.d_encoder = EncoderWithBias(config, use_one_hot_embeddings)

        self.is_training = is_training

    def call(self, inputs, **kwargs):
        de_inputs, qe_inputs = inputs
        with tf.compat.v1.variable_scope(dual_model_prefix1):
            qtype_vector1, q_bias = self.q_encoder(qe_inputs)

        with tf.compat.v1.variable_scope(dual_model_prefix2):
            qtype_vector2, d_bias = self.d_encoder(de_inputs)

        query_document_score1 = tf.reduce_sum(tf.multiply(qtype_vector1, qtype_vector2), axis=1)
        bias = tf.Variable(initial_value=0.0, trainable=True)
        query_document_score2 = query_document_score1 + bias
        query_document_score3 = query_document_score2 + q_bias
        query_document_score = query_document_score3 + d_bias
        bias2 = query_document_score2 - query_document_score1
        return {
            'query_document_score': query_document_score,
            'bias': bias2,
            'q_bias': q_bias,
            'd_bias': d_bias
        }



def define_input(max_seq_len, prefix):
    l_input_ids = tf.keras.layers.Input(shape=(max_seq_len,), dtype='int32',
                                        name=prefix + "_input_ids")
    l_input_mask = tf.keras.layers.Input(shape=(max_seq_len,), dtype='int32',
                                         name=prefix + "_input_mask")
    l_token_type_ids = tf.keras.layers.Input(shape=(max_seq_len,), dtype='int32',
                                             name=prefix + "_segment_ids")
    inputs = (l_input_ids, l_input_mask, l_token_type_ids)
    return inputs


DROP_KEYWORD = "DROP"


def name_mapping(name):
    name = name.replace("encoder_with_bias/", "SCOPE1/")
    name = name.replace("encoder_with_bias_1/", "SCOPE2/")
    name = name.replace("_embeddings/embeddings", "_embeddings")

    layer_pattern ="one_transform_layer(_)?(\d+)?"
    m = re.search(layer_pattern, name)
    if m is not None:
        if m.group(2) is not None:
            layer_no = int(m.group(2)) % 12
        else:
            layer_no = 0
        after = "layer_{}".format(layer_no)
        name = re.sub(layer_pattern, after, name)
    if name.find(DROP_KEYWORD) >= 0:
        idx = name.find(DROP_KEYWORD) + len(DROP_KEYWORD) + 1
        name = name[idx:]
    name = name.split(":")[0]
    return name


def load_qde4(save_path, model_config):
    bert_config_file = os.path.join(data_path, "bert_config.json")
    config = JsonConfig.from_json_file(bert_config_file)
    config.set_attrib("q_voca_size", model_config.q_type_voca)
    with tf.compat.v1.variable_scope(DROP_KEYWORD):
        model = QDE4(config, True, False)
        max_seq_len = model_config.max_seq_length
        qe_inputs = define_input(max_seq_len, "qe")
        de_inputs = define_input(max_seq_len, "de")
        inputs = qe_inputs, de_inputs
        out_d = model.call(inputs)
    load_stock_weights(model, save_path, name_mapping, ["optimizer", "probe_optimizer"])
    model = tf.keras.Model(inputs=inputs, outputs=out_d)
    return model

