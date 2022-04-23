from typing import List

from alignment.nli_align_path_helper import get_tfrecord_path
import tensorflow as tf

from alignment.perturbation_feature.pert_model_1d import get_dataset, build_model, weighted_MAE, precision_at_1, \
    print_last_layer_weights, binary_hinge_loss
from alignment.perturbation_feature.train_configs import get_pert_train_data_shape_1d
from alignment.related.related_answer_data_path_helper import get_model_save_path


def main():
    print("run_train_1d.py main()")
    dataset_name = "train_v1_1_2K"
    input_file = get_tfrecord_path(f"{dataset_name}")
    batch_size = 8
    shape: List[int] = get_pert_train_data_shape_1d()
    dataset = get_dataset(shape, batch_size, input_file)
    for item in dataset:
        feature, label = item
        print('x', feature['x'].shape)
        print('label', label.shape)
        print("sum(label)", tf.reduce_sum(label))


if __name__ == "__main__":
    main()