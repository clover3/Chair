from typing import NamedTuple

import tensorflow as tf

from models.transformer.bert_common_v2 import get_shape_list2
from trainer_v2.bert_for_tf2 import BertModelLayer


def vector_three_feature(v1, v2):
    concat_layer = tf.keras.layers.Concatenate()
    concat = concat_layer([v1, v2])
    sub = v1 - v2
    dot = tf.multiply(v1, v2)
    output = tf.concat([sub, dot, concat], axis=-1, name="three_feature")
    return output


class VectorThreeFeature(tf.keras.layers.Layer):
    def __init__(self):
        super(VectorThreeFeature, self).__init__()

    def call(self, inputs, *args, **kwargs):
        v1, v2 = inputs
        return vector_three_feature(v1, v2)


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


class MeanProjection(tf.keras.layers.Layer):
    def __init__(self, project_dim, prefix):
        super(MeanProjection, self).__init__()
        Dense = tf.keras.layers.Dense
        self.projector: tf.keras.layers.Dense = Dense(project_dim, activation='relu', name="{}/project".format(prefix))

    def call(self, inputs, *args, **kwargs):
        seq_p = self.projector(inputs)
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
        l_bert = BertModelLayer.from_params(bert_params, name="{}/bert".format(prefix))
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


class SplitSegmentIDLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(SplitSegmentIDLayer, self).__init__()

    def call(self, inputs, *args, **kwargs):
        rep_middle, l_input_ids, token_type_ids = inputs
        # rep_middle: [batch_size, seq_length, hidden_dim]

        def slice_segment_pad_value(segment_id_val):
            is_target_seg_mask = tf.logical_and(tf.equal(token_type_ids, segment_id_val), tf.not_equal(l_input_ids, 0))
            is_target_seg_mask = tf.cast(tf.expand_dims(is_target_seg_mask, 2), tf.float32)
            rep_middle_masked = tf.multiply(rep_middle, is_target_seg_mask)
            return rep_middle_masked

        rep_middle0 = slice_segment_pad_value(0)
        rep_middle1 = slice_segment_pad_value(1)

        return rep_middle0, rep_middle1


class SplitSegmentIDLayerWVar(tf.keras.layers.Layer):
    def __init__(self, hidden_dims):
        super(SplitSegmentIDLayerWVar, self).__init__()
        init = tf.random_normal_initializer()
        self.empty_embedding = tf.Variable(
            initial_value=init(shape=(hidden_dims,), dtype="float32"), trainable=True
        )

    def call(self, inputs, *args, **kwargs):
        rep_middle, l_input_ids, token_type_ids = inputs
        # rep_middle: [batch_size, seq_length, hidden_dim]
        batch_size, seq_length = get_shape_list2(l_input_ids)
        empty_embedding_seq = tf_tile_after_expand_dims(self.empty_embedding, [0, 1], [batch_size, seq_length, 1])

        def slice_segment_pad_value(segment_id_val):
            input_mask = tf.not_equal(l_input_ids, 0)
            is_target_seg_mask = tf.logical_and(tf.equal(token_type_ids, segment_id_val), input_mask)
            is_target_seg_mask = tf.cast(tf.expand_dims(is_target_seg_mask, 2), tf.float32)
            rep_middle_masked = tf.multiply(rep_middle, is_target_seg_mask)
            is_not_target_seq_mask = (1.0 - is_target_seg_mask)
            empty_embedding_seq_masked = tf.multiply(empty_embedding_seq, is_not_target_seq_mask)
            return rep_middle_masked + empty_embedding_seq_masked

        rep_middle0 = slice_segment_pad_value(0)
        rep_middle1 = slice_segment_pad_value(1)

        return rep_middle0, rep_middle1


class TwoLayerDense(tf.keras.layers.Layer):
    def __init__(self, hidden_size, hidden_size2, activation1='relu', activation2=tf.nn.softmax):
        super(TwoLayerDense, self).__init__()
        self.layer1 = tf.keras.layers.Dense(hidden_size, activation=activation1)
        self.layer2 = tf.keras.layers.Dense(hidden_size2, activation=activation2)

    def call(self, inputs, *args, **kwargs):
        hidden = self.layer1(inputs)
        return self.layer2(hidden)


def tf_tile_after_expand_dims(v, expand_dim_list, tile_param):
    v_ex = v
    for expand_dim in expand_dim_list:
        v_ex = tf.expand_dims(v_ex, expand_dim)
    return tf.tile(v_ex, tile_param)


class TileAfterExpandDims(tf.keras.layers.Layer):
    def __init__(self, expand_dim_raw, tile_param):
        super(TileAfterExpandDims, self).__init__()
        if type(expand_dim_raw) == int:
            self.expand_dim_list = [expand_dim_raw]
        else:
            self.expand_dim_list = expand_dim_raw
        self.tile_param = tile_param

    def call(self, inputs, *args, **kwargs):
        return tf_tile_after_expand_dims(inputs, self.expand_dim_list, self.tile_param)

