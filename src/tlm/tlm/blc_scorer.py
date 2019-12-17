import tensorflow as tf


def brutal_loss_compare(features):
    loss1 = features["loss1"]
    loss2 = features["loss2"]

    prob1 = tf.exp(-loss1)
    prob2 = tf.exp(-loss2)

    output = -(prob1 - prob2)
    return output

