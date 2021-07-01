import tensorflow as tf

from list_lib import left, right


def main():
    def normal_ce(y, p):
        v = y * tf.math.log(p) + (1-y) * tf.math.log(1-p)
        return -v

    high_bar = 0.7
    low_bar = 0.3
    def new_ce(y, p):
        is_one = tf.cast(tf.equal(y, 1), tf.float32)
        is_zero = tf.cast(tf.equal(y, 0), tf.float32)
        c = -tf.math.log(high_bar)
        losses = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(y, p)
        # if_one_loss = tf.maximum(-tf.math.log(p), c)
        # if_zero_loss = tf.maximum(-tf.math.log(1-p), c)
        # v = is_one * if_one_loss + is_zero * if_zero_loss
        losses = tf.maximum(losses, c)
        return losses

    data_points = [(1, 0.9), (1, 0.5), (0, 0.5), (0, 0.1), ]

    y_list = left(data_points)
    p_list = right(data_points)
    #
    # print("normal ce")
    # for y, p in data_points:
    #     print(y, p, normal_ce(y, p))
    #
    # print("-log 0.7", -tf.math.log(0.7))
    # print("new ce")
    # for y, p in data_points:
    #     print(y, p, new_ce(y, p))

    Y = tf.constant([[0, 1], [0, 1], [1, 0], [1, 0]])
    Y = tf.constant([1, 1, 0, 0])
    P = tf.constant([[0.9, 0.1], [0.5, 0.5], [0.5, 0.5], [0.1, 0.9]], tf.float32)
    P = tf.constant(p_list, tf.float32)
    print(new_ce(tf.expand_dims(Y, 1), tf.expand_dims(P, 1)))


def code_a():
    high_bar = 0.7
    def new_ce(y, p):
        is_one = tf.cast(tf.equal(y, 1), tf.float32)
        is_zero = tf.cast(tf.equal(y, 0), tf.float32)
        c = -tf.math.log(high_bar)
        if_one_loss = tf.maximum(-tf.math.log(p), c)
        if_zero_loss = tf.maximum(-tf.math.log(1-p), c)
        v = is_one * if_one_loss + is_zero * if_zero_loss
        return v

    data_points = [(1, 0.9), (1, 0.5), (0, 0.5), (0, 0.1), ]

    print("-log 0.7", -tf.math.log(0.7))
    print("new ce")
    for y, p in data_points:
        print(y, p, new_ce(y, p))

if __name__ == "__main__":
    code_a()
    main()