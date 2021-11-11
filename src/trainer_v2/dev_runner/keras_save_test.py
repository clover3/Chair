import os

import tensorflow as tf

from cpath import output_path
from models.keras_model.bert_keras.modular_bert import NamedLinear


def main():
    hidden_size = 6
    label_input = tf.keras.layers.Input(shape=(hidden_size,), dtype='float32', name="label_ids")
    inputs = [label_input]
    named_linear = NamedLinear(3, hidden_size)
    output = named_linear(label_input)
    model = tf.keras.models.Model(inputs=inputs, outputs=output)
    save_path = os.path.join(output_path, "model", "runs", "save_test")
    model.save(save_path)


if __name__ == "__main__":
    main()