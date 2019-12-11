import tensorflow as tf

from trainer.tf_module import p_at_1


def check_it():
    scores = tf.constant([[0.1, 0.9, 0.3], [0.3, 0.5, 0.5]])

    labels1 = tf.constant([[0,1,0], [0,1,0]])
    labels2 = tf.constant([[0, 0, 1], [0, 1, 0]])

    print(p_at_1(scores, labels1))
    print(p_at_1(scores, labels2))

    print(tf.reduce_mean(tf.cast([1,0], tf.float32)))


check_it()