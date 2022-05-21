import bert
import tensorflow as tf
from tensorflow import keras

from models.transformer.bert_common_v2 import get_shape_list2
from trainer_v2.custom_loop.modeling_common.bert_common import BERT_CLS
from trainer_v2.custom_loop.modeling_common.network_utils import vector_three_feature


class ModelConfig2Seg:
    max_seq_length1 = 200
    max_seq_length2 = 100
    num_classes = 3


class BERTSiamese:
    def __init__(self, bert_params, config: ModelConfig2Seg):
        l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")
        pooler = tf.keras.layers.Dense(bert_params.hidden_size, activation=tf.nn.tanh, name="bert/pooler/dense")
        bert_cls = BERT_CLS(l_bert, pooler)
        max_seq_len1 = config.max_seq_length1
        max_seq_len2 = config.max_seq_length2
        num_classes = config.num_classes
        pad_len = max_seq_len1 - max_seq_len2

        l_input_ids1 = keras.layers.Input(shape=(max_seq_len1,), dtype='int32', name="input_ids1")
        l_token_type_ids1 = keras.layers.Input(shape=(max_seq_len1,), dtype='int32', name="segment_ids1")

        l_input_ids2 = keras.layers.Input(shape=(max_seq_len2,), dtype='int32', name="input_ids2")
        l_token_type_ids2 = keras.layers.Input(shape=(max_seq_len2,), dtype='int32', name="segment_ids2")

        def pad(t):
            return tf.pad(t, [(0, 0), (0, pad_len)])
        l_input_ids = tf.concat([l_input_ids1, pad(l_input_ids2)], axis=0)
        l_token_type_ids = tf.concat([l_token_type_ids1, pad(l_token_type_ids2)], axis=0)

        cls = bert_cls.apply([l_input_ids, l_token_type_ids])
        batch_size, _ = get_shape_list2(l_input_ids1)
        cls_output1 = cls[:batch_size]
        cls_output2 = cls[batch_size:]

        feature_rep = vector_three_feature(cls_output1, cls_output2)
        hidden = tf.keras.layers.Dense(bert_params.hidden_size, activation='relu')(feature_rep)
        output = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)(hidden)
        inputs = (l_input_ids1, l_token_type_ids1, l_input_ids2, l_token_type_ids2)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        self.model: keras.Model = model
        self.bert_cls = bert_cls


