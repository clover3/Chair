import bert
import tensorflow as tf
from tensorflow import keras

from trainer_v2.custom_loop.modeling_common.bert_common import BERT_CLS, ModelConfig


class BERT_AssymetricDebug:
    def __init__(self, bert_params, config: ModelConfig):
        def build_bert_cls(prefix):
            l_bert = bert.BertModelLayer.from_params(bert_params, name="{}/bert".format(prefix))
            pooler = tf.keras.layers.Dense(bert_params.hidden_size, activation=tf.nn.tanh, name="{}/bert/pooler/dense".format(prefix))
            return BERT_CLS(l_bert, pooler)

        bert_cls1 = build_bert_cls("encoder1")
        bert_cls2 = build_bert_cls("encoder2")
        max_seq_len1 = config.max_seq_length
        max_seq_len2 = config.max_seq_length
        num_classes = config.num_classes

        l_input_ids1 = keras.layers.Input(shape=(max_seq_len1,), dtype='int32', name="input_ids1")
        l_token_type_ids1 = keras.layers.Input(shape=(max_seq_len1,), dtype='int32', name="segment_ids1")

        l_input_ids2 = keras.layers.Input(shape=(max_seq_len2,), dtype='int32', name="input_ids2")
        l_token_type_ids2 = keras.layers.Input(shape=(max_seq_len2,), dtype='int32', name="segment_ids2")

        cls1 = bert_cls1.apply([l_input_ids1, l_token_type_ids1])
        cls2 = bert_cls2.apply([l_input_ids2, l_token_type_ids2])

        feature_rep = cls1
        hidden = tf.keras.layers.Dense(bert_params.hidden_size, activation='relu')(feature_rep)
        output = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)(hidden)
        inputs = [l_input_ids1, l_token_type_ids1, l_input_ids2, l_token_type_ids2]
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        self.model: keras.Model = model
        self.bert_cls_list = [bert_cls1, bert_cls2]


