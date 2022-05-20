import tensorflow as tf


def vector_three_feature(v1, v2):
    concat_layer = tf.keras.layers.Concatenate()
    concat = concat_layer([v1, v2])
    sub = v1 - v2
    dot = tf.multiply(v1, v2)
    output = tf.concat([sub, dot, concat], axis=-1, name="three_feature")
    return output