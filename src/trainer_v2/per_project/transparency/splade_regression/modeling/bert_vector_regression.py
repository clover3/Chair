from typing import Dict
import tensorflow as tf
from tensorflow import keras

from trainer_v2.bert_for_tf2 import BertModelLayer
from trainer_v2.per_project.transparency.splade_regression.modeling.regression_modeling import ReluSigmoidMaxReduce


class BertVectorRegression:
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
