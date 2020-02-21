

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_ranking.python import utils

from my_tf import tf

_EPSILON = 1e-10


def compute_unreduced_loss(labels, logits):
    """See `_RankingLoss`."""
    alpha = 10.0
    is_valid = utils.is_label_valid(labels)
    labels = tf.compat.v1.where(is_valid, labels, tf.zeros_like(labels))
    logits = tf.compat.v1.where(
        is_valid, logits, -1e3 * tf.ones_like(logits) +
                          tf.reduce_min(input_tensor=logits, axis=-1, keepdims=True))

    label_sum = tf.reduce_sum(input_tensor=labels, axis=1, keepdims=True)
    nonzero_mask = tf.greater(tf.reshape(label_sum, [-1]), 0.0)
    labels = tf.compat.v1.where(nonzero_mask, labels,
                                _EPSILON * tf.ones_like(labels))
    gains = tf.pow(2., tf.cast(labels, dtype=tf.float32)) - 1.
    ranks = utils.approx_ranks(logits, alpha=alpha)
    discounts = 1. / tf.math.log1p(ranks)
    dcg = tf.reduce_sum(input_tensor=gains * discounts, axis=-1, keepdims=True)
    cost = -dcg * utils.inverse_max_dcg(labels)
    return cost, tf.reshape(tf.cast(nonzero_mask, dtype=tf.float32), [-1, 1])
