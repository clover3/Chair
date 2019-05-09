import numpy as np
import tensorflow as tf
from trainer.np_modules import *
from misc_lib import *


def batch_train(sess, batch, train_op, model):
    input, target = batch
    loss_val, _ = sess.run([model.loss, train_op],
                                feed_dict={
                                    model.x: input,
                                    model.y: target,
                                },
                           )

    return loss_val


def epoch_runner(batches, step_fn,
                 dev_fn=None, valid_freq = 1000,
                 save_fn=None, save_interval=10000,
                 shuffle=True):
    l_loss =[]
    l_acc = []
    last_save = time.time()
    if shuffle:
        np.random.shuffle(batches)
    for step_i, batch in enumerate(batches):
        if dev_fn is not None:
            if step_i % valid_freq == 0:
                dev_fn()

        if save_fn is not None:
            if time.time() - last_save > save_interval:
                save_fn()
                last_save = time.time()

        loss, acc = step_fn(batch, step_i)
        l_acc.append(acc)
        l_loss.append(loss)
    return average(l_loss), average(l_acc)


def step_runner(batches, step_fn,
                 dev_fn=None, valid_freq = 1000,
                 save_fn=None, save_interval=10000,
                 steps=999999,
                 shuffle=True):
    l_loss =[]
    l_acc = []
    last_save = time.time()
    if shuffle:
        np.random.shuffle(batches)
    for step_i, batch in enumerate(batches[:steps]):
        if dev_fn is not None:
            if step_i % valid_freq == 0:
                dev_fn()

        if save_fn is not None:
            if time.time() - last_save > save_interval:
                save_fn()
                last_save = time.time()

        loss, acc = step_fn(batch, step_i)
        l_acc.append(acc)
        l_loss.append(loss)

    return average(l_loss), average(l_acc)


# a : [batch, 2]
def cartesian_w2(a, b):
    r00 = tf.multiply(a[:,0], b[:,0]) # [None, ]
    r01 = tf.multiply(a[:,0], b[:,1])  # [None, ]
    r10 = tf.multiply(a[:, 1], b[:, 0])  # [None,]
    r11 = tf.multiply(a[:, 1], b[:, 1])  # [None,]
    r0 = tf.stack([r00, r01], axis=1)
    r1 = tf.stack([r10, r11], axis=1)
    return tf.stack([r0, r1], axis=1)


def accuracy(logits, y, axis=1):
    return tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(logits, axis=axis),
                         tf.cast(y, tf.int64)), tf.float32))


def precision(logits, y, axis=1):
    pred = tf.cast(tf.argmax(logits, axis=axis), tf.bool)
    y_int = tf.cast(y, tf.bool)
    tp = tf.logical_and(pred, tf.equal(pred,y_int))
    return tf.count_nonzero(tp) / tf.count_nonzero(pred)


def precision_b(pred, y):
    y_int = tf.cast(y, tf.bool)
    tp = tf.logical_and(pred, tf.equal(pred,y_int))
    return tf.count_nonzero(tp) / tf.count_nonzero(pred)




def recall(logits, y, axis=1):
    pred = tf.cast(tf.argmax(logits, axis=axis), tf.bool)
    y_int = tf.cast(y, tf.bool)
    tp = tf.logical_and(pred, tf.equal(pred, y_int))
    return tf.count_nonzero(tp) / tf.count_nonzero(y_int)



def recall_b(pred, y):
    y_int = tf.cast(y, tf.bool)
    tp = tf.logical_and(pred, tf.equal(pred, y_int))
    return tf.count_nonzero(tp) / tf.count_nonzero(y_int)


def f1(logits, y, axis=1):
    predictions = tf.argmax(logits, axis=axis)

    def f1_per_flag(flag_idx):
        arr_gold_pos = tf.equal(y, flag_idx)
        arr_pred_pos = tf.equal(predictions, flag_idx)
        arr_true_pos = tf.logical_and(arr_gold_pos, arr_pred_pos)

        n_true_pos = tf.count_nonzero(arr_true_pos)
        n_pred_pos = tf.count_nonzero(arr_pred_pos)
        n_gold = tf.count_nonzero(arr_gold_pos)

        prec = n_true_pos / n_pred_pos
        recall = n_true_pos / n_gold

        if (prec + recall) == 0:
            return 0
        else:
            return 2*prec*recall / (prec + recall)

    f1_favor = f1_per_flag(2)
    f1_against = f1_per_flag(1)

    return (f1_favor + f1_against) / 2


def f1_loss(logits, y):
    c01 = tf.math.less(logits[:,0], logits[:,1])
    c02 = tf.math.less(logits[:, 0], logits[:, 2])
    c21 = tf.math.less(logits[:, 2], logits[:, 1])
    c12 = tf.math.logical_not(c21)
    losses = tf.losses.softmax_cross_entropy(y, logits)
    pred1 = tf.cast(tf.math.logical_and(c01, c21), tf.float32)
    gold1 = tf.cast(tf.math.equal(y[:,1], 1), tf.float32)
    pred2 = tf.cast(tf.math.logical_and(c02, c12), tf.float32)
    gold2 = tf.cast(tf.math.equal(y[:, 2], 1), tf.float32)

    prec1 = tf.reduce_sum(pred1 * losses)
    recall1 = tf.reduce_sum(gold1 * losses)
    f1_1 = prec1 * recall1

    prec2 = tf.reduce_sum(pred2 * losses)
    recall2 = tf.reduce_sum(gold2 * losses)
    f2_1 = prec2 * recall2
    return f1_1 + f2_1


def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = tf.reduce_mean(x, axis=1, keepdims=True)
    my = tf.reduce_mean(y, axis=1, keepdims=True)
    xm, ym = x-mx, y-my
    r_num = tf.reduce_sum(tf.multiply(xm,ym), axis=1, keepdims=True)
    #r_den = tf.sqrt(tf.multiply(tf.reduce_sum(tf.square(xm), axis=1, keep_dims=True)), tf.reduce_sum(tf.square(ym), axis=1, keep_dims=True)))
    r_den = tf.sqrt(tf.reduce_sum(tf.square(xm), axis=1, keepdims=True))
    r = r_num / (r_den + 0.00000001)
    r = tf.maximum(tf.minimum(r, 1.0), -1.0)
    return -r


def cossim(v1, v2):
    v1_n = tf.nn.l2_normalize(v1, axis=1)
    v2_n = tf.nn.l2_normalize(v2, axis=1)
    return tf.losses.cosine_distance(v1_n, v2_n, axis=1)



def init_session():
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False)
    config.gpu_options.allow_growth = True

    return tf.Session(config=config)

