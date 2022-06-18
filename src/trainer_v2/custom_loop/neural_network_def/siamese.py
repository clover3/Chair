import tensorflow as tf
from tensorflow import keras

from models.transformer.bert_common_v2 import get_shape_list2
from trainer_v2.bert_for_tf2 import BertModelLayer
from trainer_v2.custom_loop.modeling_common.bert_common import BERT_CLS
from trainer_v2.custom_loop.modeling_common.network_utils import vector_three_feature


class ModelConfig2Seg:
    max_seq_length1 = 200
    max_seq_length2 = 100
    num_classes = 3


def build_siamese_inputs_apply(max_seq_len1, max_seq_len2):
    pad_len = max_seq_len1 - max_seq_len2
    l_input_ids1 = keras.layers.Input(shape=(max_seq_len1,), dtype='int32', name="input_ids1")
    l_token_type_ids1 = keras.layers.Input(shape=(max_seq_len1,), dtype='int32', name="segment_ids1")
    l_input_ids2 = keras.layers.Input(shape=(max_seq_len2,), dtype='int32', name="input_ids2")
    l_token_type_ids2 = keras.layers.Input(shape=(max_seq_len2,), dtype='int32', name="segment_ids2")
    inputs = (l_input_ids1, l_token_type_ids1, l_input_ids2, l_token_type_ids2)

    def pad(t):
        return tf.pad(t, [(0, 0), (0, pad_len)])

    l_input_ids = tf.concat([l_input_ids1, pad(l_input_ids2)], axis=0)
    l_token_type_ids = tf.concat([l_token_type_ids1, pad(l_token_type_ids2)], axis=0)
    batch_size, _ = get_shape_list2(l_input_ids1)
    return batch_size, inputs, l_input_ids, l_token_type_ids


class BERTSiamese:
    def __init__(self, bert_params, config: ModelConfig2Seg):
        l_bert = BertModelLayer.from_params(bert_params, name="bert")
        pooler = tf.keras.layers.Dense(bert_params.hidden_size, activation=tf.nn.tanh, name="bert/pooler/dense")
        bert_cls = BERT_CLS(l_bert, pooler)
        num_classes = config.num_classes
        batch_size, inputs, l_input_ids, l_token_type_ids = \
            build_siamese_inputs_apply(config.max_seq_length1, config.max_seq_length2)

        cls = bert_cls.apply([l_input_ids, l_token_type_ids])
        cls_output1 = cls[:batch_size]
        cls_output2 = cls[batch_size:]

        feature_rep = vector_three_feature(cls_output1, cls_output2)
        hidden = tf.keras.layers.Dense(bert_params.hidden_size, activation='relu')(feature_rep)
        output = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)(hidden)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        self.model: keras.Model = model
        self.bert_cls = bert_cls


class BERTSiameseL:
    def __init__(self, bert_params, config: ModelConfig2Seg):
        l_bert = BertModelLayer.from_params(bert_params, name="bert")
        pooler = tf.keras.layers.Dense(bert_params.hidden_size, activation=tf.nn.tanh, name="bert/pooler/dense")
        bert_cls = BERT_CLS(l_bert, pooler)
        num_classes = config.num_classes
        batch_size, inputs, l_input_ids, l_token_type_ids = \
            build_siamese_inputs_apply(config.max_seq_length1, config.max_seq_length2)

        cls = bert_cls.apply([l_input_ids, l_token_type_ids])
        cls_output1 = cls[:batch_size]
        cls_output2 = cls[batch_size:]

        feature_rep = vector_three_feature(cls_output1, cls_output2)
        output = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)(feature_rep)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        self.model: keras.Model = model
        self.bert_cls = bert_cls


class ModelConfig2SegProject(ModelConfig2Seg):
    max_seq_length1 = 200
    max_seq_length2 = 100
    num_classes = 3
    project_dim = 4 * 728


class ModelConfigTwoProject(ModelConfig2Seg):
    max_seq_length1 = 200
    max_seq_length2 = 100
    num_classes = 3
    project_dim = 4 * 728
    project_dim2 = 728


class BERTSiameseMean:
    def __init__(self, bert_params, config: ModelConfig2SegProject):
        Dense = tf.keras.layers.Dense
        l_bert = BertModelLayer.from_params(bert_params, name="bert")

        batch_size, inputs, l_input_ids, l_token_type_ids =\
            build_siamese_inputs_apply(config.max_seq_length1, config.max_seq_length2)
        seq_output = l_bert([l_input_ids, l_token_type_ids])
        seq_p = Dense(config.project_dim, activation='relu')(seq_output)
        seq_m = tf.reduce_mean(seq_p, axis=1)
        seq_rep = seq_m

        rep1 = seq_rep[:batch_size]
        rep2 = seq_rep[batch_size:]

        feature_rep = vector_three_feature(rep1, rep2)
        hidden = tf.keras.layers.Dense(bert_params.hidden_size, activation='relu')(feature_rep)
        output = tf.keras.layers.Dense(config.num_classes, activation=tf.nn.softmax)(hidden)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        self.l_bert = l_bert
        self.model: keras.Model = model


class BERTSiameseMC:
    def __init__(self, bert_params, config: ModelConfig2SegProject):
        Dense = tf.keras.layers.Dense
        l_bert = BertModelLayer.from_params(bert_params, name="bert")

        batch_size, inputs, l_input_ids, l_token_type_ids = \
            build_siamese_inputs_apply(config.max_seq_length1, config.max_seq_length2)
        seq_output = l_bert([l_input_ids, l_token_type_ids])
        seq_p = Dense(config.project_dim, activation='relu')(seq_output)
        seq_m = tf.reduce_mean(seq_p, axis=1)
        seq_rep = seq_m

        rep1 = seq_rep[:batch_size]
        rep2 = seq_rep[batch_size:]

        concat_layer = tf.keras.layers.Concatenate()
        feature_rep = concat_layer([rep1, rep2])

        hidden = tf.keras.layers.Dense(bert_params.hidden_size, activation='relu')(feature_rep)
        output = tf.keras.layers.Dense(config.num_classes, activation=tf.nn.softmax)(hidden)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        self.l_bert = l_bert
        self.model: keras.Model = model



class ModelConfig200_200(ModelConfig2Seg):
    max_seq_length1 = 200
    max_seq_length2 = 200
    num_classes = 3
    project_dim = 4 * 728
