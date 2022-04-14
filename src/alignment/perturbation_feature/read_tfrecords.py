import tensorflow as tf
from typing import List, Iterable, Callable, Dict, Tuple, Set

from alignment.nli_align_path_helper import get_tfrecord_path
from alignment.perturbation_feature.train_configs import get_pert_train_data_shape
from trainer_v2.input_fn_common import format_dataset


def perturbation_feature(shape):
    FixedLenFeature = tf.io.FixedLenFeature
    x_data_size = reduce_multiply(shape)
    y_data_size = shape[0] * shape[1]
    features = {
        "x": FixedLenFeature([x_data_size], tf.float32),
        "y": FixedLenFeature([y_data_size], tf.float32),
    }
    return features


def reduce_multiply(shape):
    x_data_size = 1
    for s in shape:
        x_data_size *= s
    return x_data_size


def main():
    dataset_name = "train"
    scorer_name = "lexical_v1"
    input_file = get_tfrecord_path(f"{dataset_name}_{scorer_name}")
    batch_size = 4
    shape: List[int] = get_pert_train_data_shape()
    dataset = format_dataset(perturbation_feature(shape), True,
                   [input_file], batch_size)

    print(dataset)


if __name__ == "__main__":
    main()