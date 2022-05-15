import logging

import tensorflow as tf

from trainer_v2.chair_logging import c_log


def modify_tf_logger():
    print('modify_tf_logger')
    c_log = logging.getLogger('tensorflow')
    c_log.setLevel(logging.INFO)
    format_str = '%(levelname)s\t%(name)s \t%(asctime)s %(message)s'
    formatter = logging.Formatter(format_str,
                                  datefmt='%m-%d %H:%M:%S',
                                  )
    for ch in c_log.handlers:
        ch.setFormatter(formatter)

    c_log.addHandler(ch)
    c_log.info("tf logging")

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
    predictions = model(x_train[:1]).numpy()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5)



def main():
    c_log.info("hi")
    import logging
    tf.get_logger().setLevel(logging.ERROR)

    # modify_tf_logger()
    do_tf_thing()
    return NotImplemented


if __name__ == "__main__":
    main()