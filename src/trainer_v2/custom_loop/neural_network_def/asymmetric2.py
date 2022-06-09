from typing import NamedTuple

import tensorflow as tf
from tensorflow import keras

from trainer_v2.bert_for_tf2 import BertModelLayer
from trainer_v2.custom_loop.modeling_common.bert_common import load_stock_weights, define_bert_input
from trainer_v2.custom_loop.modeling_common.network_utils import VectorThreeFeature, \
    MeanProjectionEnc
from trainer_v2.custom_loop.neural_network_def.asymmetric import BERTAsymmetricProjectMean
from trainer_v2.custom_loop.neural_network_def.inner_network import ClassificationModelIF
from trainer_v2.custom_loop.neural_network_def.segmented_enc import StackedInputMapper
from trainer_v2.custom_loop.neural_network_def.siamese import ModelConfig2SegProject


class AsymmetricMeanPool(ClassificationModelIF):
    def __init__(self, inner_network=BERTAsymmetricProjectMean):
        super(AsymmetricMeanPool, self).__init__()
        self.inner_network = inner_network

    def build_model(self, bert_params, model_config):
        network = self.inner_network(bert_params, model_config)
        self.network = network

    def get_keras_model(self):
        return self.network.model

    def init_checkpoint(self, init_checkpoint):
        l_bert1, l_bert2 = self.network.l_bert_list
        load_stock_weights(l_bert1, init_checkpoint, n_expected_restore=197)
        load_stock_weights(l_bert2, init_checkpoint, n_expected_restore=197)


class BERTEvenSegmented(ClassificationModelIF):
    def __init__(self, combine_local_decisions_layer):
        super(BERTEvenSegmented, self).__init__()
        self.combine_local_decisions_layer = combine_local_decisions_layer

    def build_model(self, bert_params, config: ModelConfig2SegProject):
        num_window = 2
        encoder1 = MeanProjectionEnc(bert_params, config.project_dim, "encoder1")
        encoder2 = MeanProjectionEnc(bert_params, config.project_dim, "encoder2")

        num_classes = config.num_classes

        max_seq_len1 = config.max_seq_length1
        max_seq_len2 = config.max_seq_length2
        l_input_ids1, l_token_type_ids1 = define_bert_input(config.max_seq_length1, "1")
        l_input_ids2, l_token_type_ids2 = define_bert_input(config.max_seq_length2, "2")

        # [batch_size, dim]
        rep1 = encoder1.call([l_input_ids1, l_token_type_ids1])

        window_length = int(max_seq_len2 / num_window)
        inputs_for_seg2 = [l_input_ids2, l_token_type_ids2]

        mapper = StackedInputMapper(encoder2, max_seq_len2, window_length)
        # [batch_size, num_window, dim]
        rep2_stacked = mapper(inputs_for_seg2)

        rep1_ = tf.expand_dims(rep1, 1)
        rep1_stacked = tf.tile(rep1_, [1, num_window, 1])

        # [batch_size, num_window, dim2 ]
        vtf = VectorThreeFeature()
        feature_rep = vtf((rep1_stacked, rep2_stacked))
        hidden = tf.keras.layers.Dense(bert_params.hidden_size, activation='relu')(feature_rep)
        local_decisions = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)(hidden)
        self.local_decisions = local_decisions
        combine_local_decisions = self.combine_local_decisions_layer()
        self.cld = combine_local_decisions
        output = combine_local_decisions(local_decisions)
        inputs = (l_input_ids1, l_token_type_ids1, l_input_ids2, l_token_type_ids2)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        self.model: keras.Model = model
        self.l_bert_list = [encoder1.l_bert, encoder2.l_bert]

    def get_keras_model(self):
        return self.model

    def init_checkpoint(self, init_checkpoint):
        l_bert1, l_bert2 = self.l_bert_list
        load_stock_weights(l_bert1, init_checkpoint, n_expected_restore=197)
        load_stock_weights(l_bert2, init_checkpoint, n_expected_restore=197)


class BERTEvenSegmentedWCallback(BERTEvenSegmented):
    def __init__(self, combine_local_decisions_layer):
        super(BERTEvenSegmentedWCallback, self).__init__(combine_local_decisions_layer)

    def callback(self, arg):
        self.cld.callback(arg)


# Projection + Mean+ Concat
class BERTAsymmetricPMC(ClassificationModelIF):
    def __init__(self):
        super(BERTAsymmetricPMC, self).__init__()

    def build_model(self, bert_params, config: ModelConfig2SegProject):
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
            l_bert = BertModelLayer.from_params(bert_params, name="{}/bert".format(prefix))
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

    def get_keras_model(self):
        return self.model

    def init_checkpoint(self, init_checkpoint):
        l_bert1, l_bert2 = self.l_bert_list
        load_stock_weights(l_bert1, init_checkpoint, n_expected_restore=197)
        load_stock_weights(l_bert2, init_checkpoint, n_expected_restore=197)


