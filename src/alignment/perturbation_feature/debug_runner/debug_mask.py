from typing import List

from alignment.nli_align_path_helper import get_tfrecord_path
from alignment.perturbation_feature.train_configs import get_pert_train_data_shape
from alignment.perturbation_feature.pert_model_2d import weighted_MAE, precision_at_1, build_model, get_dataset, \
    get_is_not_padding
import tensorflow as tf


def main():
    print("debug_mask main()")
    dataset_name = "train_sm"
    scorer_name = "lexical_v1"
    input_file = get_tfrecord_path(f"{dataset_name}_{scorer_name}")
    batch_size = 4
    shape: List[int] = get_pert_train_data_shape()
    dataset = get_dataset(shape, batch_size, input_file)

    for batch in dataset:
        x, y = batch
        is_not_padding = get_is_not_padding(y)
        y0 = y[0]
        for i in range(10):
            row = tf.reshape(y0[i], [-1]).numpy()
            print(row.tolist())
        # print('y', y)
        # print('is_not_padding', is_not_padding)
        # print('precision_at1', precision_at_1(y, y))
        break

if __name__ == "__main__":
    main()