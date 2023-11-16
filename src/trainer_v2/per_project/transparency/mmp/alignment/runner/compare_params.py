import sys

import tensorflow as tf
import numpy as np


from list_lib import list_equal
from trainer_v2.chair_logging import c_log


def get_model_variable_d(model_path, name_map):
    model = tf.keras.models.load_model(model_path, compile=False)
    mapping = {}
    weights = model.weights
    for weight in weights:
        key = weight.name
        tensor = weight
        mapping[name_map(key)] = tensor
    return mapping


def compare_param(model_path1, model_path2):
    def name_map(key):
        return key

    mapping1 = get_model_variable_d(model_path1, name_map)
    mapping2 = get_model_variable_d(model_path2, name_map)

    for k in mapping1:
        if k in mapping2:
            print(k)
            v1 = mapping1[k].numpy()
            v2 = mapping2[k].numpy()

            if not list_equal(v1.shape, v2.shape):
                c_log.warning("Variable shape does not match {} != {}".format(v1.shape, v2.shape))
                continue
            err = np.sum(v1 - v2)
            if not np.abs(err) < 1e-8:
                c_log.warning("Variable values differs: error={}".format(err))


def main():
    compare_param(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()