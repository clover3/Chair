from collections import Counter


import warnings
import tensorflow as tf
from trainer.tf_module import *
from models.transformer.hyperparams import Hyperparams
from task.MLM import TransformerLM
from task.classification import TransformerClassifier

from data_generator.stance import stance_detection
from log import log
import path
import os

# Experiment is the most outside module.
# This module can reference any module in the system.


def batch_train(sess, batch, train_op, model):
    input, target = batch
    loss_val, _ = sess.run([model.loss, train_op],
                                feed_dict={
                                    model.x: input,
                                    model.y: target,
                                })
    return loss_val


class Experiment:
    def __init__(self, hparam):
        self.reg_lambda = 1e-1 # TODO apply it
        self.global_step = tf.Variable(0, name='global_step',
                                       trainable=False)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.hparam = hparam
        self.save_interval = 10 * 60
        self.log = log.train_logger()
        self.model_dir = path.model_path
        self.sess = None

    @staticmethod
    def init_sess():
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
        config.gpu_options.allow_growth = True

        return tf.Session(config=config)

    def get_train_op(self, loss):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.hparam.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
        train_op = optimizer.minimize(loss, global_step=self.global_step)
        return train_op


    def temp_saver(self):
        self.save_model("interval")

    def train_stance(self):
        print("train_stance")
        valid_freq = 1000
        voca_size = stance_detection.vocab_size
        num_classes = 3
        task = TransformerClassifier(self.hparam, voca_size, num_classes, True)
        train_op = self.get_train_op(task.loss)


        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())

        data = stance_detection.DataLoader()
        train_batches = get_batches(data.get_train_data(), self.hparam.batch_size)
        dev_batches = get_batches(data.get_dev_data(), self.hparam.batch_size)


        def train_fn(batch, step_i):
            loss_val = batch_train(self.sess, batch, train_op, task)
            self.log.debug("[Train] Step {0} loss={1:.04f}".format(step_i, loss_val))
            self.merged = tf.summary.merge_all()
            return loss_val, 0

        def valid_fn():
            loss_list = []
            acc_list = []
            for batch in dev_batches:
                input, target = batch
                loss_val, acc = self.sess.run([task.loss, task.acc],
                                    feed_dict={
                                        task.x: input,
                                        task.y: target,
                                    })
                loss_list.append(loss_val)
                acc_list.append(acc)

            self.merged = tf.summary.merge_all()
            avg_loss, avg_acc = average(loss_list), average(acc_list)
            self.log.info("[Dev] : acc={0:.02f} loss={1:.04f}".format(avg_acc, avg_loss))
            return

        num_epoch = Hyperparams.num_epochs
        print("Start Training")
        for i in range(num_epoch):
            loss, _ = epoch_runner(train_batches, train_fn,
                                   valid_fn, valid_freq,
                                   self.temp_saver, self.save_interval)
            self.log.info("[Train] Epoch {} Done. Loss={}".format(i, loss))

        self.save_model("after_train")

    def train_lm(self):
        print("train_lm")
        self.sess = self.init_sess()
        valid_freq = 1000
        voca_size = NotImplemented


        task = TransformerLM(self.hparam, voca_size, True)
        train_op = self.get_train_op(task.loss)

        train_data = data.get_train_data()

        def train_fn(batch, step_i):
            loss_val = batch_train(self.sess, batch, train_op, task)
            self.log.debug("Step {0} train loss={1:.04f}".format(step_i, loss_val))
            self.merged = tf.summary.merge_all()
            return loss_val, 0

        step_fn = self.inject_saver(train_fn, self.temp_saver, self.save_interval)

        dev_batch = NotImplemented

        def valid_fn():
            loss_list = []
            for batch in dev_batch:
                input, target = batch
                loss_val, = self.sess.run([task.loss, ],
                                    feed_dict = {
                                        task.x: input,
                                        task.y: target,
                                    })

                self.log.info("Validation : loss={0:.04f}".format(loss_val))
                loss_list.append(loss_val)

            self.merged = tf.summary.merge_all()

        batches = get_batches((X,Y), self.hparam.batch_size)
        loss, _ = epoch_runner(batches, step_fn, valid_fn, valid_freq)

        self.save_model("after_train")

    def save_model(self, name):
        save_dir = os.path.join(self.model_dir, 'runs', name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        path = os.path.join(save_dir, "model")
        self.saver.save(self.sess, path, global_step=self.global_step)
        self.log.info("Model saved at {} - {}".format(path, self.global_step))


