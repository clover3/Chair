from typing import List

from alignment.nli_align_path_helper import get_tfrecord_path
import tensorflow as tf

from alignment.perturbation_feature.pert_model_1d import get_dataset, build_model, weighted_MAE, precision_at_1, \
    print_last_layer_weights, hinge_loss
from alignment.perturbation_feature.train_configs import get_pert_train_data_shape_1d


def main():
    print("run_train_1d.py main()")
    dataset_name = "train_head_v1_1"
    input_file = get_tfrecord_path(f"{dataset_name}")
    batch_size = 8
    shape: List[int] = get_pert_train_data_shape_1d()
    dataset = get_dataset(shape, batch_size, input_file)
    running_option = "mlp"
    print("Model uses {} layer".format(running_option))
    model = build_model(shape, running_option)
    loss_fn = hinge_loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(optimizer=optimizer,
                  metrics=[precision_at_1],
                  loss=loss_fn)
    model.fit(dataset, batch_size=batch_size, epochs=100)
    print("run_train - main() - Done")
    print_last_layer_weights(model)


if __name__ == "__main__":
    main()