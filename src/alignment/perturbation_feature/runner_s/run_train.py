from typing import List

from alignment.nli_align_path_helper import get_tfrecord_path
from alignment.perturbation_feature.train_configs import get_pert_train_data_shape
from alignment.perturbation_feature.train_fns import weighted_MAE, precision_at_1, build_model, get_dataset
import tensorflow as tf
import numpy as np


def print_weights(model):
    dense_layer = model.layers[-1]
    print(dense_layer)
    w, b = dense_layer.get_weights()
    print(np.reshape(w, [9, 3]))


def main():
    print("run_train main()")
    dataset_name = "train_2k"
    scorer_name = "lexical_v1"
    input_file = get_tfrecord_path(f"{dataset_name}_{scorer_name}")
    batch_size = 8
    shape: List[int] = get_pert_train_data_shape()
    dataset = get_dataset(shape, batch_size, input_file)
    model = build_model(shape)
    loss_fn = weighted_MAE
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(optimizer=optimizer,
                  metrics=[precision_at_1],
                  loss=loss_fn)
    model.fit(dataset, batch_size=batch_size, epochs=100)
    print("run_train - main() - Done")
    print_weights(model)


if __name__ == "__main__":
    main()