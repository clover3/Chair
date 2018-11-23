import numpy as np
import tensorflow as tf

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


def recall(logits, y, axis=1):
    pred = tf.cast(tf.argmax(logits, axis=axis), tf.bool)
    y_int = tf.cast(y, tf.bool)
    tp = tf.logical_and(pred, tf.equal(pred, y_int))
    return tf.count_nonzero(tp) / tf.count_nonzero(y_int)


def label2logit(label, size):
    r = np.zeros([size,])
    r[label] = 1
    return r


def get_batches(data, batch_size):
    X, Y = data
    # data is fully numpy array here
    step_size = int((len(Y) + batch_size - 1) / batch_size)
    new_data = []
    for step in range(step_size):
        x = []
        y = []
        for i in range(batch_size):
            idx = step * batch_size + i
            if idx >= len(Y):
                break
            x.append(np.array(X[idx]))
            y.append(Y[idx])
        if len(y) > 0:
            new_data.append((np.array(x), np.array(y)))

    return new_data

def get_batches_ex(data, batch_size, n_inputs):
    # data is fully numpy array here
    step_size = int((len(data) + batch_size - 1) / batch_size)
    new_data = []
    for step in range(step_size):
        b_unit = [list() for i in range(n_inputs)]

        for i in range(batch_size):
            idx = step * batch_size + i
            if idx >= len(data):
                break
            for input_i in range(n_inputs):
                b_unit[input_i].append(data[idx][input_i])
        if len(b_unit[0]) > 0:
            batch = [np.array(b_unit[input_i]) for input_i in range(n_inputs)]
            new_data.append(batch)

    return new_data



def init_session():
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False)
    config.gpu_options.allow_growth = True

    return tf.Session(config=config)

