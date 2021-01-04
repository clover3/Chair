import tensorflow as tf


def get_pos_only_weight_param(shape, name):
    output_weights = tf.compat.v1.get_variable(
        name, shape,
        initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02)
    )
    return tf.sigmoid(output_weights)