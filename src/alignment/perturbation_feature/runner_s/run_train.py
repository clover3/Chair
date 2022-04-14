import tensorflow as tf

from typing import List, Iterable, Callable, Dict, Tuple, Set

from alignment.nli_align_path_helper import get_tfrecord_path
from alignment.perturbation_feature.read_tfrecords import perturbation_feature, reduce_multiply
from alignment.perturbation_feature.train_configs import get_pert_train_data_shape
from trainer_v2.input_fn_common import format_dataset




def main():
    print("run_train main()")
    dataset_name = "train"
    scorer_name = "lexical_v1"
    input_file = get_tfrecord_path(f"{dataset_name}_{scorer_name}")
    batch_size = 4
    shape: List[int] = get_pert_train_data_shape()
    dataset = get_dataset(shape, batch_size, input_file)
    model = build_model(shape)

    def loss_fn(y, prediction):
        is_padding = tf.less(tf.reduce_sum(tf.reduce_sum(y, axis=3), axis=2), 1e-6)
        n_valid = tf.reduce_sum(tf.cast(is_padding, tf.float32), axis=1)
        error_array = tf.abs(y - prediction)
        error_per_seg1 = tf.reduce_mean(tf.reduce_mean(error_array, axis=3), axis=2)
        avg_error = tf.reduce_sum(error_per_seg1) * (1 / n_valid)
        return tf.reduce_mean(avg_error)

    model.compile(optimizer='Adam',
                  loss=loss_fn)
    model.fit(dataset, batch_size=batch_size)
    print("run_train - main() - Done")


def build_model(shape):
    x_shape = [shape[0], shape[1], shape[2] * shape[3]]
    x = tf.keras.Input(shape=x_shape, name='x', dtype=tf.float32)
    dense_layer = tf.keras.layers.Dense(1)
    prediction = dense_layer(x)
    model = tf.keras.Model(inputs={'x': x},
                           outputs=[prediction])
    return model


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


if __name__ == "__main__":
    main()