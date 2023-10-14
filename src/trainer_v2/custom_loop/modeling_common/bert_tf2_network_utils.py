from typing import NamedTuple

import tensorflow as tf

from trainer_v2.bert_for_tf2 import BertModelLayer


def get_two_projected_mean_encoder(bert_params, project_dim):
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
        projector = Dense(project_dim, activation='relu', name="{}/project".format(prefix))
        return Encoder(l_bert, projector)

    encoder1 = build_encoder("encoder1")
    encoder2 = build_encoder("encoder2")
    return encoder1, encoder2


class MeanProjectionEnc(tf.keras.layers.Layer):
    def __init__(self, bert_params, project_dim, prefix):
        super(MeanProjectionEnc, self).__init__()
        Dense = tf.keras.layers.Dense
        self.l_bert: tf.keras.layers.Layer = BertModelLayer.from_params(bert_params, name="{}/bert".format(prefix))
        self.projector: tf.keras.layers.Dense = Dense(project_dim, activation='relu', name="{}/project".format(prefix))

    def call(self, inputs, *args, **kwargs):
        seq_out = self.l_bert(inputs)
        seq_p = self.projector(seq_out)
        seq_m = tf.reduce_mean(seq_p, axis=1)
        return seq_m