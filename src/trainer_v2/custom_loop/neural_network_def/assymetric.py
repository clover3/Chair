from typing import NamedTuple

import bert
import tensorflow as tf
from tensorflow import keras

from trainer_v2.custom_loop.modeling_common.bert_common import BERT_CLS
from trainer_v2.custom_loop.modeling_common.network_utils import vector_three_feature, get_two_projected_mean_encoder
from trainer_v2.custom_loop.neural_network_def.siamese import ModelConfig2SegProject


class ModelConfig2Seg:
    max_seq_length1 = 200
    max_seq_length2 = 100
    num_classes = 3


class BERTAssymetric:
    def __init__(self, bert_params, config: ModelConfig2Seg):
        def build_bert_cls(prefix):
            l_bert = bert.BertModelLayer.from_params(bert_params, name="{}/bert".format(prefix))
            pooler = tf.keras.layers.Dense(bert_params.hidden_size, activation=tf.nn.tanh, name="{}/bert/pooler/dense".format(prefix))
            return BERT_CLS(l_bert, pooler)

        bert_cls1 = build_bert_cls("encoder1")
        bert_cls2 = build_bert_cls("encoder2")
        max_seq_len1 = config.max_seq_length1
        max_seq_len2 = config.max_seq_length2
        num_classes = config.num_classes

        l_input_ids1 = keras.layers.Input(shape=(max_seq_len1,), dtype='int32', name="input_ids1")
        l_token_type_ids1 = keras.layers.Input(shape=(max_seq_len1,), dtype='int32', name="segment_ids1")

        l_input_ids2 = keras.layers.Input(shape=(max_seq_len2,), dtype='int32', name="input_ids2")
        l_token_type_ids2 = keras.layers.Input(shape=(max_seq_len2,), dtype='int32', name="segment_ids2")

        cls1 = bert_cls1.apply([l_input_ids1, l_token_type_ids1])
        cls2 = bert_cls2.apply([l_input_ids2, l_token_type_ids2])

        feature_rep = vector_three_feature(cls1, cls2)
        hidden = tf.keras.layers.Dense(bert_params.hidden_size, activation='relu')(feature_rep)
        output = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)(hidden)
        inputs = (l_input_ids1, l_token_type_ids1, l_input_ids2, l_token_type_ids2)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        self.model: keras.Model = model
        self.bert_cls_list = [bert_cls1, bert_cls2]


class BERTAsymmetricProjectMean:
    def __init__(self, bert_params, config: ModelConfig2SegProject):
        encoder1, encoder2 = get_two_projected_mean_encoder(bert_params, config.project_dim)

        max_seq_len1 = config.max_seq_length1
        max_seq_len2 = config.max_seq_length2
        num_classes = config.num_classes

        l_input_ids1 = keras.layers.Input(shape=(max_seq_len1,), dtype='int32', name="input_ids1")
        l_token_type_ids1 = keras.layers.Input(shape=(max_seq_len1,), dtype='int32', name="segment_ids1")

        l_input_ids2 = keras.layers.Input(shape=(max_seq_len2,), dtype='int32', name="input_ids2")
        l_token_type_ids2 = keras.layers.Input(shape=(max_seq_len2,), dtype='int32', name="segment_ids2")

        rep1 = encoder1.apply([l_input_ids1, l_token_type_ids1])
        rep2 = encoder2.apply([l_input_ids2, l_token_type_ids2])

        feature_rep = vector_three_feature(rep1, rep2)
        hidden = tf.keras.layers.Dense(bert_params.hidden_size, activation='relu')(feature_rep)
        output = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)(hidden)
        inputs = (l_input_ids1, l_token_type_ids1, l_input_ids2, l_token_type_ids2)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        self.model: keras.Model = model
        self.l_bert_list = [encoder1.l_bert, encoder2.l_bert]


# Projection + Mean+ Concat
class BERTAsymmetricPMC:
    def __init__(self, bert_params, config: ModelConfig2SegProject):
        Dense = tf.keras.layers.Dense

        class Encoder(NamedTuple):
            l_bert: tf.keras.layers.Layer
            projector: tf.keras.layers.Dense

            def apply(self, inputs):
                seq_out = self.l_bert(inputs)
                seq_p = self.projector(seq_out)
                seq_m = tf.reduce_mean(seq_p, axis=1)
                return seq_m

        def build_encoder(prefix) -> Encoder:
            l_bert = bert.BertModelLayer.from_params(bert_params, name="{}/bert".format(prefix))
            projector = Dense(config.project_dim, activation='relu', name="{}/project".format(prefix))
            return Encoder(l_bert, projector)

        encoder1 = build_encoder("encoder1")
        encoder2 = build_encoder("encoder2")
        max_seq_len1 = config.max_seq_length1
        max_seq_len2 = config.max_seq_length2
        num_classes = config.num_classes

        l_input_ids1 = keras.layers.Input(shape=(max_seq_len1,), dtype='int32', name="input_ids1")
        l_token_type_ids1 = keras.layers.Input(shape=(max_seq_len1,), dtype='int32', name="segment_ids1")

        l_input_ids2 = keras.layers.Input(shape=(max_seq_len2,), dtype='int32', name="input_ids2")
        l_token_type_ids2 = keras.layers.Input(shape=(max_seq_len2,), dtype='int32', name="segment_ids2")

        rep1 = encoder1.apply([l_input_ids1, l_token_type_ids1])
        rep2 = encoder2.apply([l_input_ids2, l_token_type_ids2])

        concat_layer = tf.keras.layers.Concatenate()
        feature_rep = concat_layer([rep1, rep2])

        hidden = tf.keras.layers.Dense(bert_params.hidden_size, activation='relu')(feature_rep)
        output = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)(hidden)
        inputs = (l_input_ids1, l_token_type_ids1, l_input_ids2, l_token_type_ids2)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        self.model: keras.Model = model
        self.l_bert_list = [encoder1.l_bert, encoder2.l_bert]
