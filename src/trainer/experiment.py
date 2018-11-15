from collections import Counter

import tensorflow as tf
from trainer.tf_module import *
from task.MLM import TransformerLM
import os


def batch_train(sess, train_op, batch, model):
    input, target = batch
    loss_val, _ = sess.run([model.loss, train_op],
                                feed_dict={
                                    model.x: input,
                                    model.y: target,
                                })
    return loss_val

class Experiment:
    def __init__(self, hparam):

        self.reg_lambda = 1e-1
        self.global_step = tf.Variable(0, name='global_step',
                                       trainable=False)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.hparam = hparam
        self.save_interval = 10 * 60
        self.log = NotImplemented
        self.model_dir = NotImplemented

    @staticmethod
    def init_sess():
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
        config.gpu_options.allow_growth = True

        return tf.Session(config=config)


    def get_train_op(self, loss):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.hparam.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
        train_op = self.optimizer.minimize(loss, global_step=self.global_step)
        return train_op

    @staticmethod
    def inject_saver(train_fn, saver, time_interval):
        last_save = -1
        def step_fn(batch, i):
            global last_save
            train_fn(batch, i)
            if time.time() - last_save > time_interval:
                saver()
                last_save = time.time()
        return step_fn

    def temp_saver(self):
        self.save_model("interval")

    def train_lm(self, data_getter):
        self.sess = self.init_sess()
        valid_freq = 1000
        voca_size = NotImplemented

        task = TransformerLM(self.hparam, voca_size, True)
        train_op = self.get_train_op(task.loss)

        X, Y = data_getter()

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
                loss_val, _ = self.sess.run([task.loss, train_op],
                                    feed_dict={
                                        task.x: input,
                                        task.y: target,
                                    })

                self.log.info("Validation : loss={0:.04f}".format(loss_val))
                loss_list.append(loss_val)

            self.merged = tf.summary.merge_all()
            return average(loss_list), 0

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


