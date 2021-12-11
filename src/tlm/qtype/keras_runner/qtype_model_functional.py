import os

import tensorflow as tf

from cpath import data_path
from models.keras_model.bert_keras.modular_bert import BertLayer
from models.transformer.bert_common_v2 import create_initializer
from tlm.model.base import BertModel
from tlm.model.dual_model_common import dual_model_prefix1, dual_model_prefix2
from tlm.model_cnfig import JsonConfig
from tlm.qtype.keras_runner.qtype_models import define_input
from tlm.qtype.model_server.mmd_console import load_stock_weights
from tlm.qtype.qde_model_fn import qtype_modeling_single_mlp


### THIS CODE DOES NOT WORK
def qde4_functional(inputs, model_config, use_one_hot_embeddings, is_training=False):
    if not is_training:
        model_config.hidden_dropout_prob = 0.0
        model_config.attention_probs_dropout_prob = 0.0

    qe_inputs, de_inputs = inputs
    def single_bias_model(config, vector):
        dense = tf.keras.layers.Dense(1, activation=tf.keras.activations.tanh,
                                      kernel_initializer=create_initializer(config.initializer_range))
        v = dense(vector)
        return tf.reshape(v, [-1])

    with tf.compat.v1.variable_scope(dual_model_prefix1):
        qe_input_ids, qe_input_mask, qe_token_type_ids = qe_inputs
        bert_layer = BertLayer(model_config, True, True)
        sequence_output = bert_layer.call((qe_input_ids, qe_input_mask, qe_token_type_ids))
        last_layer = sequence_output[-1]
        first_token_tensor = tf.squeeze(last_layer[:, 0:1, :], axis=1)
        pooled1 = first_token_tensor
        model_1 = BertModel(
            config=model_config,
            is_training=is_training,
            input_ids=qe_input_ids,
            input_mask=qe_input_mask,
            token_type_ids=qe_token_type_ids,
            use_one_hot_embeddings=use_one_hot_embeddings,
                                               )
        pooled1 = model_1.get_pooled_output() # [batch_size * 2, hidden_size]
        qtype_vector1 = qtype_modeling_single_mlp(model_config, pooled1)  # [batch_size * 2, qtype_length]
        q_bias = single_bias_model(model_config, pooled1)
    return qtype_vector1
    with tf.compat.v1.variable_scope(dual_model_prefix2):
        de_input_ids, de_input_mask, de_token_type_ids = de_inputs
        model_2 = BertModel(
            config=model_config,
            is_training=is_training,
            input_ids=de_input_ids,
            input_mask=de_input_mask,
            token_type_ids=de_token_type_ids,
            use_one_hot_embeddings=use_one_hot_embeddings,
                                                 )
        pooled2 = model_2.get_pooled_output()
        qtype_vector2 = qtype_modeling_single_mlp(model_config, pooled2)
        d_bias = single_bias_model(model_config, pooled2)

    query_document_score1 = tf.reduce_sum(tf.multiply(qtype_vector1, qtype_vector2), axis=1)
    # bias = tf.Variable(initial_value=0.0, trainable=True)
    # print('bias', bias)
    # query_document_score2 = query_document_score1 + bias
    query_document_score2 = query_document_score1
    query_document_score3 = query_document_score2 + q_bias
    query_document_score = query_document_score3 + d_bias
    return query_document_score
    # return {
    #     'query_document_score': query_document_score,
    #     'bias': bias2,
    #     'q_bias': q_bias,
    #     'd_bias': d_bias
    # }


def name_mapping(name):
    name = name.split(":")[0]
    name = name.replace("LayerNorm", "layer_normalization")
    name = name.replace("/embeddings", "")
    return name


def load_qde4(save_path, model_config):
    bert_config_file = os.path.join(data_path, "bert_config.json")
    config = JsonConfig.from_json_file(bert_config_file)
    config.set_attrib("q_voca_size", model_config.q_type_voca)
    max_seq_len = model_config.max_seq_length
    qe_inputs = define_input(max_seq_len, "qe")
    de_inputs = define_input(max_seq_len, "de")
    inputs = qe_inputs, de_inputs
    print("Building network")
    outputs = qde4_functional(inputs, config, True, False)
    print("Model def")
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    print("load_stock_weights")
    load_stock_weights(model, save_path, name_mapping, ["optimizer"])
    print("load_qde4 done")
    return model

### THIS CODE DOES NOT WORK