import functools
import os

import tensorflow as tf

from cpath import output_path


def do_tf_thing():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    steps_per_epoch = 10000
    steps_per_loop = 1
    batch_size = 2
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric_fn = [functools.partial(
        tf.keras.metrics.SparseCategoricalAccuracy,
        'accuracy',
        dtype=tf.float32)]
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=[fn() for fn in metric_fn],
                  steps_per_execution=steps_per_loop
                  )
    metric_fn = [metric_fn]

    summary_dir = os.path.join(output_path, 'summaries')
    summary_callback = tf.keras.callbacks.TensorBoard(summary_dir,
                                                      write_graph=False,
                                                      update_freq=1
                                                      )
    custom_callbacks = [summary_callback]
    history = model.fit(x_train, y_train,
              steps_per_epoch=steps_per_epoch,
              validation_data=None,
              batch_size=2,
              epochs=5,
              callbacks=custom_callbacks
              )


def main():
    do_tf_thing()


if __name__ == "__main__":
    main()