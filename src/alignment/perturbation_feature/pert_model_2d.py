import tensorflow as tf

from alignment.perturbation_feature.read_tfrecords import perturbation_feature
from trainer_v2.input_fn_common import format_dataset


def weighted_MAE(y, prediction):
    is_padding_seg1 = tf.less(tf.reduce_sum(tf.reduce_sum(y, axis=3), axis=2), 1e-6)
    is_padding_seg2 = tf.less(tf.reduce_sum(tf.reduce_sum(y, axis=3), axis=1), 1e-6)
    is_valid_seg1 = tf.cast(tf.logical_not(is_padding_seg1), tf.float32)
    is_valid_seg2 = tf.cast(tf.logical_not(is_padding_seg2), tf.float32)
    is_valid_mask = tf.matmul(tf.expand_dims(is_valid_seg1, 2), tf.expand_dims(is_valid_seg2, 1))
    label_3d = tf.squeeze(y, axis=3)
    pred_3d = tf.squeeze(prediction, axis=3)

    n_valid = tf.reduce_sum(tf.reduce_sum(is_valid_mask, axis=1), axis=1)
    error_array = tf.abs(label_3d - pred_3d)
    error_array = error_array * is_valid_mask
    error_per_seg1 = tf.reduce_sum(error_array, axis=2)
    avg_error = tf.reduce_sum(error_per_seg1) * (1 / n_valid)
    return tf.reduce_mean(avg_error)


def binary_ce(y, prediction):
    label = tf.less(0.5, y)
    is_padding_f = get_is_not_padding(y)
    n_valid = tf.reduce_sum(is_padding_f, axis=1)
    label_3d = tf.squeeze(label, axis=3)
    pred_3d = tf.squeeze(prediction, axis=3)
    loss_arr = tf.keras.losses.BinaryCrossentropy()(label_3d, pred_3d)
    loss_per_seg1 = tf.reduce_mean(loss_arr, axis=2)
    avg_error = tf.reduce_sum(error_per_seg1) * (1 / n_valid)
    return tf.reduce_mean(avg_error)


def precision_at_1(y, prediction):
    prediction_3d = tf.squeeze(prediction, 3)
    y_3d = tf.squeeze(y, 3)
    max_seg2_idx = tf.argmax(prediction_3d, axis=2)  # [None, 256]
    max_seg2_idx = tf.expand_dims(max_seg2_idx, 2)
    label_at_max = tf.gather(y_3d, max_seg2_idx, axis=2, batch_dims=2)  # [None, 256]
    label_at_max = tf.squeeze(label_at_max, 2)

    is_padding_f = get_is_not_padding(y)
    n_valid = tf.reduce_sum(is_padding_f, axis=1)
    avg_score = tf.reduce_sum(label_at_max * is_padding_f, axis=1) * (1 / n_valid)
    return avg_score


def get_is_not_padding(y):
    is_padding = tf.less(tf.reduce_sum(tf.reduce_sum(y, axis=3), axis=2), 1e-6)
    is_not_padding = tf.logical_not(is_padding)
    is_not_padding_f = tf.cast(is_not_padding, tf.float32)
    return is_not_padding_f


# Options
#  1)Logits as feature

def build_model(shape):
    x_shape = [shape[0], shape[1], shape[2] * shape[3]]
    x = tf.keras.Input(shape=x_shape, name='x', dtype=tf.float32)
    prediction = network_inner(shape, x)
    model = tf.keras.Model(inputs={'x': x},
                           outputs=[prediction])
    return model


def network_inner(shape, x):
    x_structure = tf.reshape(x, [-1] + shape)
    probs = tf.nn.softmax(x_structure, axis=3)
    probs_flat = tf.reshape(probs, [-1] + [shape[0], shape[1], shape[2] * shape[3]])
    dense_layer = tf.keras.layers.Dense(1)
    prediction = dense_layer(probs_flat)
    return prediction


def get_dataset(shape, batch_size, input_file):
    dataset = format_dataset(perturbation_feature(shape), True,
                             [input_file], False)
    x_shape = [shape[0], shape[1], shape[2] * shape[3]]
    y_shape = [shape[0], shape[1], 1]

    def reshape_xy(f):
        return {
                   'x': tf.reshape(f['x'], x_shape),
               }, tf.reshape(f['y'], y_shape)

    dataset = dataset.map(reshape_xy)
    dataset = dataset.batch(batch_size)
    return dataset