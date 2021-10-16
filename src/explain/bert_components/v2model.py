import os

import tensorflow as tf

from cpath import output_path


def main():
    model_path = os.path.join(output_path, "model", "runs", "ex_run4")
    model = tf.keras.models.load_model(model_path)


    return NotImplemented


if __name__ == "__main__":
    main()