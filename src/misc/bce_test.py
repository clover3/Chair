import tensorflow as tf
from keras.losses import BinaryCrossentropy


def main():
    y_gold = tf.constant([[0], [1], [1], [0], [1], [1]])
    y_pred = tf.constant([[0.1], [0.9], [1.5], [-1], [1e-7], [-1]])
    losses = BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)\
        (y_gold, y_pred)
    print(losses)


if __name__ == "__main__":
    main()