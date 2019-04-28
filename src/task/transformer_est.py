from __future__ import print_function


import tensorflow as tf

from models.transformer.transformer import transformer_encode
from models.losses import *
from models.transformer import bert
from trainer import tf_module


class LanguageModel:
    def __init__(self, voca_size):
        self.voca_size = voca_size

    def predict(self, enc, Y, is_training):
        logits = tf.layers.dense(enc, self.voca_size)
        preds = tf.to_int32(tf.argmax(logits, axis=-1))
        istarget = tf.to_float(tf.not_equal(Y, 0))
        acc = tf.reduce_sum(tf.to_float(tf.equal(preds, Y)) * istarget) / (tf.reduce_sum(istarget))
        label_smoothing = 0.1
        if is_training:
            # Loss
            loss_list, weight = padded_cross_entropy(logits, Y, label_smoothing,
                                                     reduce_sum=False)
            self.loss = tf.reduce_mean(loss_list)
            return preds, self.loss
        else:
            return preds


class Classification:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def predict(self, enc, Y, is_training):
        feature_loc = 0
        logits = tf.layers.dense(enc[:,feature_loc,:], self.num_classes, name="cls_dense")
        labels = tf.one_hot(Y, self.num_classes)
        preds = tf.to_int32(tf.argmax(logits, axis=-1))
        self.acc = tf_module.accuracy(logits, Y)
        self.logits = logits

        if is_training:
            loss_arr = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits,
                labels=labels)
            self.loss = tf.reduce_mean(loss_arr)
            return preds, self.loss
        else:
            return preds


class Transformer:
    def __init__(self, hp, voca_size, task):
        self.hp = hp
        self.voca_size = voca_size
        self.task = task
        self.dtype = tf.float32

    def model_fn(self, mode, features, labels, params):

        # train mode: required loss and train_op
        # eval mode: required loss
        # predict mode: required predictions
        metrics = {}
        train_op = None
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
            predictions, loss = self.predict(features, labels, True)
            train_op = self.get_train_op(loss)
            metrics = {"accuracy" : self.task.acc }
            tf.summary.scalar("accuracy", self.task.acc[1])
        else:
            predictions = self.predict(features, labels, False)
            loss = None

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics,
            predictions={"prediction": predictions})

    def predict(self, X, Y, is_training):
        self.enc = transformer_encode(X, self.hp, self.voca_size, is_training)
        return self.task.predict(self.enc, Y, is_training)

    def get_train_op(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return train_op


