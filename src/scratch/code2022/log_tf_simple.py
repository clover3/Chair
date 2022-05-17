import logging

import tensorflow as tf
from official import nlp

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
    c_log.info("main 1")
    import logging
    root_logger = logging.getLogger()
    print("c_log.handlers", c_log.handlers)
    print("root_logger handler", root_logger.handlers)
    tf.get_logger().setLevel(logging.ERROR)
    tf_logger = logging.getLogger('tensorflow')
    c_log.info("main 2")
    tf_logger.propagate = False
    tf_logger.info("tf logger say somthing")
    print("root_logger handler", root_logger.handlers)
    optimizer = nlp.optimization.create_optimizer(1, num_train_steps=1000, num_warmup_steps=1)
    print("tf_logger handler", tf_logger.handlers)
    tf_logger.info("tf logger say somthing2")
    c_log.info("main 3")
    print("root_logger handler", root_logger.handlers)
    tf_logger.info("tf logger say somthing3")
    c_log.info("main 4")
    c_log.info("main 5")




if __name__ == "__main__":
    main()