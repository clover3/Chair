from collections import Counter


import warnings
import tensorflow as tf
from trainer.tf_module import *
from models.transformer.hyperparams import Hyperparams
from task.MLM import TransformerLM
from task.PairLM import TransformerPairLM
from task.classification import TransformerClassifier
from sklearn.feature_extraction.text import CountVectorizer

from data_generator.stance import stance_detection
from data_generator.mask_lm import enwiki
from data_generator import shared_setting
from task.metrics import stance_f1
from log import log
import path
import os
import pickle
from models.baselines import svm
from tensorflow.python.client import timeline


# Experiment is the most outside module.
# This module can reference any module in the system.

class Experiment:
    def __init__(self, hparam):
        #self.reg_lambda = 1e-1
        self.global_step = tf.Variable(0, name='global_step',
                                       trainable=False)
        self.saver = None
        self.hparam = hparam
        self.save_interval = 10 * 60
        self.log = log.train_logger()
        self.model_dir = path.model_path
        self.sess = None

    @staticmethod
    def init_sess():
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False
                                )
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
        max_sequence= 140
        stance_data = stance_detection.DataLoader(max_sequence)

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




    def train_stance(self, voca_size, stance_data, preload_id = None):
        print("Experiment.train_stance()")
        valid_freq = 1000
        task = TransformerClassifier(self.hparam, voca_size, stance_detection.num_classes, True)
        train_op = self.get_train_op(task.loss)

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())

        if preload_id is not None:
            name = preload_id[0]
            id = preload_id[1]
            self.load_model(name, id)

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

    def train_lm_inf(self, exp_config):
        print("train_lm")
        valid_freq = exp_config.valid_freq
        step_per_epoch = exp_config.step_per_epoch
        save_interval = exp_config.save_interval
        setting = shared_setting.Enwiki2Stance()
        voca_size = setting.vocab_size

        data = enwiki.DataLoader(self.hparam.seq_max, setting)

        task = TransformerLM(self.hparam, voca_size, True)
        train_op = self.get_train_op(task.loss)
        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())

        train_data = data.get_train_generator()
        test_data = data.get_test_generator()

        def pack2batch(data_source):
            X = []
            Y = []
            for i in range(self.hparam.batch_size):
                x, y = data_source.__next__()
                X.append(x)
                Y.append(y)
            return np.array(X), np.array(Y)

        def train_fn(dummy, step_i):
            batch = pack2batch(train_data)
            loss_val = batch_train(self.sess, batch, train_op, task)
            self.log.debug("Step {0} train loss={1:.04f}".format(step_i, loss_val))
            self.merged = tf.summary.merge_all()
            return loss_val, 0


        def get_dev_data():
            num_batches = 5
            for i in range(num_batches):
                yield pack2batch(test_data)

        dev_batch = list(get_dev_data())

        def valid_fn():
            loss_list = []
            for batch in dev_batch:
                input, target = batch
                loss_val, = self.sess.run([task.loss, ],
                                    feed_dict = {
                                        task.x: input,
                                        task.y: target,
                                    })


                loss_list.append(loss_val)
            self.log.info("Validation : loss={0:.04f}".format(average(loss_list)))
            self.merged = tf.summary.merge_all()


        def save_fn():
            self.save_model("LM")

        step_fn = train_fn
        last_save = time.time()
        for step_i in range(step_per_epoch):
            if step_i % valid_freq == 0:
                valid_fn()

            if save_fn is not None:
                if time.time() - last_save > save_interval:
                    save_fn()
                    last_save = time.time()

            step_fn(None, step_i)

        self.save_model("after_train")

    def train_lm_batch(self, exp_config, data):
        print("train_lm_batch")
        valid_freq = exp_config.valid_freq
        save_interval = exp_config.save_interval

        task = TransformerLM(self.hparam, data.voca_size, True)
        train_op = self.get_train_op(task.loss)
        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())

        use_cache = False
        if not use_cache:
            print("Generating data")
            train_data = data.get_train_generator()
            train_batches = get_batches(zip(*list(train_data)), self.hparam.batch_size)

            test_data = data.get_test_generator()
            test_batches = get_batches(zip(*list(test_data)), self.hparam.batch_size)
            pickle.dump((train_batches, test_batches), open("batch_cache.pickle", "wb"))
        else:
            train_batches, test_batches = pickle.load(open("batch_cache.pickle", "rb"))

        def train_fn(batch, step_i):
            loss_val = batch_train(self.sess, batch, train_op, task)
            self.log.debug("Step {0} train loss={1:.04f}".format(step_i, loss_val))
            return loss_val, 0

        def valid_fn():
            loss_list = []
            for batch in test_batches:
                input, target = batch
                print(input)
                print(target)
                loss_val, = self.sess.run([task.loss, ],
                                          feed_dict = {
                                              task.x: input,
                                              task.y: target,
                                          })


                loss_list.append(loss_val)
            self.log.info("Validation : loss={0:.04f}".format(average(loss_list)))
            self.merged = tf.summary.merge_all()

        def save_fn():
            self.save_model(exp_config.name)

        for epoch_i in range(exp_config.num_epoch):
            epoch_runner(train_batches, train_fn, valid_fn, valid_freq, save_fn, save_interval)

        self.save_model(exp_config.name+"_final")


    def train_pair_lm(self, exp_config, data):
        print("train_lm_ex")
        valid_freq = exp_config.valid_freq
        save_interval = exp_config.save_interval

        task = TransformerPairLM(self.hparam, data.voca_size, 2, True)
        train_op = self.get_train_op(task.loss)
        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())

        use_cache = False
        if not use_cache:
            print("Generating data")
            n_inputs = 3

            train_data = data.get_train_generator()
            train_batches = get_batches_ex(list(train_data), self.hparam.batch_size, n_inputs)

            test_data = data.get_test_generator()
            test_batches = get_batches_ex(list(test_data), self.hparam.batch_size, n_inputs)
            pickle.dump((train_batches, test_batches), open("batch_cache.pickle", "wb"))
        else:
            train_batches, test_batches = pickle.load(open("batch_cache.pickle", "rb"))

        def batch2feed_dict(batch):
            x, y_seq, y_cls = batch
            feed_dict = {
                task.x: x,
                task.y_seq: y_seq,
                task.y_cls: y_cls,
            }
            return feed_dict

        def train_fn(batch, step_i):
            loss_val, _ = self.sess.run([task.loss, train_op],
                                        feed_dict=batch2feed_dict(batch)
                                        )
            self.log.debug("Step {0} train loss={1:.04f}".format(step_i, loss_val))
            return loss_val, 0

        def valid_fn():
            loss_list = []
            for batch in test_batches:
                loss_val, = self.sess.run([task.loss, ],
                                          feed_dict= batch2feed_dict(batch)
                                          )
                loss_list.append(loss_val)
            self.log.info("Validation : loss={0:.04f}".format(average(loss_list)))
            self.merged = tf.summary.merge_all()

        def save_fn():
            self.save_model(exp_config.name)

        for epoch_i in range(exp_config.num_epoch):
            epoch_runner(train_batches, train_fn, valid_fn, valid_freq, save_fn, save_interval)

        self.save_model(exp_config.name+"_final")

    def save_model(self, name):
        run_dir = os.path.join(self.model_dir, 'runs')
        save_dir = os.path.join(run_dir, name)

        exist_or_mkdir(self.model_dir)
        exist_or_mkdir(run_dir)
        exist_or_mkdir(save_dir)

        path = os.path.join(save_dir, "model")
        if self.saver is None:
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        ret = self.saver.save(self.sess, path, global_step=self.global_step)
        self.log.info("Model saved at {} - {}".format(path, ret))


    def load_model(self, name, id):
        run_dir = os.path.join(self.model_dir, 'runs')
        save_dir = os.path.join(run_dir, name)
        path = os.path.join(save_dir, "model-{}".format(id))

        variables = tf.contrib.slim.get_variables_to_restore()

        def condition(v):
            return v.name.split('/')[0] == 'encoder'

        variables_to_restore = [v for v in variables if condition(v)]
        print("Restoring:")
        for v in variables_to_restore:
            print(v)

        self.loader = tf.train.Saver(variables_to_restore, max_to_keep=1)
        self.loader.restore(self.sess, path)