from typing import List

from alignment.nli_align_path_helper import get_tfrecord_path
import tensorflow as tf

from alignment.perturbation_feature.pert_model_1d import get_dataset, build_model, weighted_MAE, precision_at_1, \
    print_last_layer_weights, binary_hinge_loss, split_val
from alignment.perturbation_feature.train_configs import get_pert_train_data_shape_1d
from alignment.related.related_answer_data_path_helper import get_model_save_path


def main():
    print("run_train_1d.py main()")
    dataset_name = "train_v1_1_2K"
    input_file = get_tfrecord_path(f"{dataset_name}")
    batch_size = 8
    shape: List[int] = get_pert_train_data_shape_1d()
    dataset = get_dataset(shape, batch_size, input_file)
    train, val = split_val(dataset, int(2000 / 8))
    running_option = "linear"
    save_name = dataset_name
    print("Model uses {} layer".format(running_option))
    model = build_model(shape, running_option)
    loss_fn = binary_hinge_loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(optimizer=optimizer,
                  metrics=[precision_at_1],
                  loss=loss_fn)
    model.fit(train, validation_data=val, batch_size=batch_size, epochs=20)
    print("run_train - main() - Done")
    print_last_layer_weights(model)
    model.save(get_model_save_path(save_name))


if __name__ == "__main__":
    main()