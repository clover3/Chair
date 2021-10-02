import numpy as np
import tensorflow as tf

num_feature = 100

def main():
    target_function = tf.math.cos
    model = get_model()
    optimizer = tf.keras.optimizers.Adam(1e-4)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.MAE)
    x_train = tf.random.uniform([1000, num_feature])
    y_train = target_function(x_train)
    x_dev = tf.random.uniform([10, num_feature])
    y_dev = target_function(x_dev)

    model.fit(x_train, y_train, epochs=100)
    model.summary()
    print(model.evaluate(x_dev, y_dev))


def get_model():
    x_input = tf.keras.Input([num_feature])
    x = tf.expand_dims(x_input, -1)
    conv = tf.keras.layers.Conv1D(32, 1)
    print(x)
    W = tf.Variable(np.array([1.0]), dtype=tf.float32)
    # y = conv(x)
    y = tf.multiply(W, x)
    bias = tf.random.normal([1, num_feature])
    y = tf.reduce_sum(y, axis=2)
    y = y + bias
    model = tf.keras.Model(inputs=[x_input], outputs=[y])
    return model


if __name__ == "__main__":
    main()
