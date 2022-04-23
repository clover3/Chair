from typing import List

from alignment.nli_align_path_helper import get_tfrecord_path
import tensorflow as tf

from alignment.perturbation_feature.pert_model_1d import get_dataset, \
    precision_at_1, binary_hinge_loss, build_model, print_last_layer_weights_w_shape
from alignment.perturbation_feature.train_configs import get_pert_train4_data_shape
from alignment.related.related_answer_data_path_helper import get_model_save_path


def main():
    print("run_train_1d.py main()")
    dataset_name = "train_4_5K"
    input_file = get_tfrecord_path(f"{dataset_name}")
    batch_size = 8
    shape: List[int] = get_pert_train4_data_shape()
    dataset = get_dataset(shape, batch_size, input_file)
    running_option = "linear"
    save_name = f"{dataset_name}_{running_option}"
    print("Model uses {} layer".format(running_option))
    model = build_model(shape, running_option)
    loss_fn = binary_hinge_loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(optimizer=optimizer,
                  metrics=[precision_at_1],
                  loss=loss_fn)
    model.fit(dataset, batch_size=batch_size, epochs=20)
    print("run_train - main() - Done")
    for batch in dataset:
        feature, label = batch
        y = model.predict(feature)
        print(y)
        break
    print_last_layer_weights_w_shape(model, shape)
    model.save(get_model_save_path(save_name))



if __name__ == "__main__":
    main()