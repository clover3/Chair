from typing import NamedTuple

import bert
import tensorflow as tf

from models.transformer.bert_common_v2 import get_shape_list2


def vector_three_feature(v1, v2):
    concat_layer = tf.keras.layers.Concatenate()
    concat = concat_layer([v1, v2])
    sub = v1 - v2
    dot = tf.multiply(v1, v2)
    output = tf.concat([sub, dot, concat], axis=-1, name="three_feature")
    return output


class VectorThreeFeature(tf.keras.layers.Layer):
    def call(self, inputs, *args, **kwargs):
        v1, v2 = inputs
        return vector_three_feature(v1, v2)


class MeanProjectionEnc(tf.keras.layers.Layer):
    def __init__(self, bert_params, project_dim, prefix):
        super(MeanProjectionEnc, self).__init__()
        Dense = tf.keras.layers.Dense
        self.l_bert: tf.keras.layers.Layer = bert.BertModelLayer.from_params(bert_params, name="{}/bert".format(prefix))
        self.projector: tf.keras.layers.Dense = Dense(project_dim, activation='relu', name="{}/project".format(prefix))

    def call(self, inputs, *args, **kwargs):
        seq_out = self.l_bert(inputs)
        seq_p = self.projector(seq_out)
        seq_m = tf.reduce_mean(seq_p, axis=1)
        return seq_m


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
        l_bert = bert.BertModelLayer.from_params(bert_params, name="{}/bert".format(prefix))
        projector = Dense(project_dim, activation='relu', name="{}/project".format(prefix))
        return Encoder(l_bert, projector)

    encoder1 = build_encoder("encoder1")
    encoder2 = build_encoder("encoder2")
    return encoder1, encoder2


def split_stack_input(input_ids_like_list,
                total_seq_length: int,
                window_length: int,
                ):
    # e.g. input_id_like_list[0] shape is [8, 250 * 4],  it return [8 * 4, 250]
    num_window = int(total_seq_length / window_length)
    batch_size, _ = get_shape_list2(input_ids_like_list[0])

    def r2to3(arr):
        return tf.reshape(arr, [batch_size, num_window, window_length])

    return list(map(r2to3, input_ids_like_list))


