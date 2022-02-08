import tensorflow as tf

# alpha.shape = (batch_size, n_seq, n_seq)
from models.transformer.bert_common_v2 import get_shape_list


@tf.custom_gradient
def cap_layer(param):
    y = tf.minimum(tf.maximum(param, 0), 1)
    def grad(dy):
        return dy

    return y, grad


def hard_concrete(log_alpha, n_sample):
    eps = 1e-20
    batch_size, n_seq1, n_seq2 = get_shape_list(log_alpha)
    shape = [batch_size, n_sample, n_seq1, n_seq2] # (batch_size, n_sample, n_seq, n_seq)
    u = tf.random.uniform(shape, minval=0, maxval=1)
    l = tf.math.log(u + eps) - tf.math.log(1 - u + eps)
    temperature = 0.2
    log_alpha_ex = tf.expand_dims(log_alpha, 1)
    y = log_alpha_ex + l
    gamma = -0.1
    zeta = 1.1
    z_hat = tf.nn.sigmoid(y / temperature)
    z_hat_shift = z_hat * (zeta - gamma) + gamma
    z = cap_layer(z_hat_shift)
    # z = tf.minimum(tf.maximum(z_hat_shift, 0), 1)
    return z


def hard_concrete_inf(log_alpha):
    return tf.cast(tf.less(0.0, log_alpha), tf.float32)


def get_sample_accuracy(base_logits, masked_logits):
    gold_pred = tf.argmax(base_logits, axis=2)
    masked_pred = tf.argmax(masked_logits, axis=2)
    sample_accuracy = tf.reduce_mean(tf.cast(tf.equal(gold_pred, masked_pred), tf.float32), axis=1)
    return sample_accuracy