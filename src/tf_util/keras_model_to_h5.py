import sys

import h5py
import tensorflow as tf


def convert_keras_model_to_h5(tf_model_path, save_path):
    model = tf.keras.models.load_model(tf_model_path, compile=False)

    fs = h5py.File(save_path, 'w')
    weights = model.weights
    for weight in weights:
        key = weight.name
        tensor = weight
        fs.create_dataset(key, data=tensor.numpy())
    fs.close()


if __name__ == "__main__":
    convert_keras_model_to_h5(sys.argv[1], sys.argv[2])

