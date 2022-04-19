import tensorflow as tf
from typing import List, Iterable, Callable, Dict, Tuple, Set

from alignment.perturbation_feature.read_tfrecords import reduce_multiply
from trainer_v2.input_fn_common import format_dataset


def weighted_MAE(y, prediction):
    # y.shape [batch, seq_len, 1]
    # label_2d.shape [batch, seq_len]
    label_2d = tf.squeeze(y, axis=2)

    # if 0, it should be padding
    is_padding = get_is_padding(label_2d)
    is_valid_mask = tf.cast(tf.logical_not(is_padding), tf.float32)
    pred_2d = tf.squeeze(prediction, axis=2)

    n_valid = tf.reduce_sum(is_valid_mask, axis=1)
    error_array = tf.abs(label_2d - pred_2d)
    error_array = error_array * is_valid_mask
    error_per_inst = tf.reduce_sum(error_array, axis=1) * (1 / n_valid)
    return tf.reduce_mean(error_per_inst)


def get_is_padding(label_2d):
    return tf.logical_and(tf.less(-1e-6, label_2d), tf.less(label_2d, 1e-6))


def precision_at_1(y, prediction):
    prediction_2d = tf.squeeze(prediction, 2)
    y_2d = tf.squeeze(y, 2)
    max_seg_idx = tf.expand_dims(tf.argmax(prediction_2d, axis=1), 1)  # [None, ]
    label_at_max = tf.gather(y_2d, max_seg_idx, axis=1, batch_dims=1)  # [None, 256]
    label_at_max = tf.squeeze(label_at_max, 1)
    avg_score = tf.reduce_mean(label_at_max)
    return avg_score



def perturbation_feature(shape):
    FixedLenFeature = tf.io.FixedLenFeature
    x_data_size = reduce_multiply(shape)
    y_data_size = shape[0]
    features = {
        "x": FixedLenFeature([x_data_size], tf.float32),
        "y": FixedLenFeature([y_data_size], tf.float32),
    }
    return features


def build_model(shape, modeling_option):
    x_shape = [shape[0], shape[1] * shape[2]]
    x = tf.keras.Input(shape=x_shape, name='x', dtype=tf.float32)
    prediction = network_inner(shape, x, modeling_option)
    model = tf.keras.Model(inputs={'x': x},
                           outputs=[prediction])
    return model


def network_inner(shape, x, modeling_option):
    x_structure = tf.reshape(x, [-1] + shape)
    probs = tf.nn.softmax(x_structure, axis=2)
    probs_flat = tf.reshape(probs, [-1] + [shape[0], shape[1] * shape[2]])
    if modeling_option == "linear":
        dense_layer = tf.keras.layers.Dense(1)
        prediction = dense_layer(probs_flat)
    elif modeling_option == "mlp":
        h_size = shape[1] * shape[2]
        h = tf.keras.layers.Dense(h_size, activation="relu")(probs_flat)
        dense_layer = tf.keras.layers.Dense(1)
        prediction = dense_layer(h)
    else:
        raise ValueError(modeling_option)
    return prediction



def get_dataset(shape, batch_size, input_file):
    dataset = format_dataset(perturbation_feature(shape), True,
                             [input_file], False)
    x_shape = [shape[0], shape[1] * shape[2]]
    y_shape = [shape[0], 1]

    def reshape_xy(f):
        return {
                   'x': tf.reshape(f['x'], x_shape),
               }, tf.reshape(f['y'], y_shape)

    dataset = dataset.map(reshape_xy)
    dataset = dataset.batch(batch_size)
    return dataset