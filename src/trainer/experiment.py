from collections import Counter


import warnings
import tensorflow as tf
from trainer.tf_module import *
from models.transformer.hyperparams import Hyperparams
from task.MLM import TransformerLM
from task.classification import TransformerClassifier
from sklearn.feature_extraction.text import CountVectorizer

from data_generator.stance import stance_detection
from task.metrics import stance_f1
from log import log
import path
import os
from models.baselines import svm

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


    def stance_baseline(self):
        print("Experiment.stance_baseline()")
        stance_data = stance_detection.DataLoader()

        # TODO majority
        stance_data.load_train_data()
        stance_data.load_test_data()
        label_count = Counter()

        for entry in stance_data.train_data_raw:
            label_count[entry["label"]] += 1

        common_label, _ = label_count.most_common(1)[0]
        pred_major = np.array([common_label] * len(stance_data.dev_data_raw))

        # TODO word unigram SVM

        train_x = [entry["inputs"] for entry in stance_data.train_data_raw]
        train_y = [entry["label"] for entry in stance_data.train_data_raw]
        dev_x = [entry["inputs"] for entry in stance_data.dev_data_raw]
        dev_y = [entry["label"] for entry in stance_data.dev_data_raw]

        pred_svm_uni = svm.train_svm_and_test(CountVectorizer(), train_x, train_y, dev_x)
        pred_svm_ngram = svm.train_svm_and_test(svm.NGramFeature(), train_x, train_y, dev_x)

        print("Major : {0:.02f}".format(stance_f1(pred_major, dev_y)))
        print("unigram svm: {0:.02f}".format(stance_f1(pred_svm_uni, dev_y)))
        print("ngram svm: {0:.02f}".format(stance_f1(pred_svm_ngram, dev_y)))

        test_x = [entry["inputs"] for entry in stance_data.test_data_raw]
        test_y = [entry["label"] for entry in stance_data.test_data_raw]

        pred_major_test = np.array([common_label] * len(stance_data.test_data_raw))
        pred_svm_uni_test = svm.train_svm_and_test(CountVectorizer(), train_x, train_y, test_x)
        pred_svm_ngram_test = svm.train_svm_and_test(svm.NGramFeature(), train_x, train_y, test_x)

        print("<Test>")
        print("Major : {0:.02f}".format(stance_f1(pred_major_test, test_y)))
        print("unigram svm: {0:.02f}".format(stance_f1(pred_svm_uni_test, test_y)))
        print("ngram svm: {0:.02f}".format(stance_f1(pred_svm_ngram_test, test_y)))


    def train_stance(self):
        print("Experiment.train_stance()")
        valid_freq = 1000
        voca_size = stance_detection.vocab_size
        task = TransformerClassifier(self.hparam, voca_size, stance_detection.num_classes, True)
        train_op = self.get_train_op(task.loss)


        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())

        stance_data = stance_detection.DataLoader()
        train_batches = get_batches(stance_data.get_train_data(), self.hparam.batch_size)
        dev_batches = get_batches(stance_data.get_dev_data(), self.hparam.batch_size)

        def train_fn(batch, step_i):
            loss_val = batch_train(self.sess, batch, train_op, task)
            self.log.debug("[Train] Step {0} loss={1:.04f}".format(step_i, loss_val))
            self.merged = tf.summary.merge_all()
            return loss_val, 0

        def valid_fn():
            loss_list = []
            acc_list = []
            logits_list = []
            gold_list = []
            for batch in dev_batches:
                input, target = batch
                loss_val, acc, logits = self.sess.run([task.loss, task.acc, task.logits],
                                    feed_dict={
                                        task.x: input,
                                        task.y: target,
                                    })
                loss_list.append(loss_val)
                acc_list.append(acc)
                logits_list.append(logits)
                gold_list.append(target)

            logits = np.concatenate(logits_list, axis=0)
            pred = np.argmax(logits, axis=1)
            f1 = stance_f1(pred, np.concatenate(gold_list, axis=0))
            self.merged = tf.summary.merge_all()
            avg_loss, avg_acc = average(loss_list), average(acc_list)
            self.log.info("[Dev] F1={0:.02f} Acc={1:.02f} loss={2:.04f}".format(f1, avg_acc, avg_loss))
            return

        num_epoch = Hyperparams.num_epochs
        print("Start Training")
        for i in range(num_epoch):
            loss, _ = epoch_runner(train_batches, train_fn,
                                   valid_fn, valid_freq,
                                   self.temp_saver, self.save_interval)
            self.log.info("[Train] Epoch {} Done. Loss={}".format(i, loss))

        valid_fn()
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
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        run_dir = os.path.join(self.model_dir, 'runs')
        if not os.path.exists(run_dir):
            os.mkdir(run_dir)
        save_dir = os.path.join(run_dir, name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        path = os.path.join(save_dir, "model")
        self.saver.save(self.sess, path, global_step=self.global_step)
        self.log.info("Model saved at {} - {}".format(path, self.global_step))


