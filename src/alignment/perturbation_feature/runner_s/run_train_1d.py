from typing import List

from alignment.nli_align_path_helper import get_tfrecord_path
import tensorflow as tf
import numpy as np

from alignment.perturbation_feature.pert_model_1d import get_dataset, build_model, weighted_MAE, precision_at_1
from alignment.perturbation_feature.train_configs import get_pert_train_data_shape_1d


def print_weights(model):
    dense_layer = model.layers[-1]
    print(dense_layer)
    w, b = dense_layer.get_weights()
    print(np.reshape(w, [9, 3]))


def main():
    print("run_train_1d.py main()")
    dataset_name = "train_2k_row"
    input_file = get_tfrecord_path(f"{dataset_name}")
    batch_size = 8
    shape: List[int] = get_pert_train_data_shape_1d()
    dataset = get_dataset(shape, batch_size, input_file)
    running_option = "mlp"
    print("Model uses {} layer".format(running_option))
    model = build_model(shape, running_option)
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