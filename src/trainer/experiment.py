from collections import Counter

import random
import warnings
import tensorflow as tf
from concurrent.futures import ProcessPoolExecutor, wait

from trainer.tf_module import *
from trainer.promise import *
from models.transformer.hyperparams import Hyperparams
from task.MLM import TransformerLM
from task.PairLM import TransformerPairLM
from task.AuxLM import AuxLM
from task.PairFeature import PairFeature, PairFeatureClassification
from task.classification import TransformerClassifier
from task.consistent_classification import ConsistentClassifier
from task.aux_classification import AuxClassification
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, roc_curve

from data_generator.stance import stance_detection
from data_generator.mask_lm import enwiki
from data_generator import shared_setting
from data_generator.NLI.nli import eval_explain
from data_generator.adhoc.ws import *
from data_generator.data_parser.trec import *
from data_generator.data_parser import controversy
from trainer.queue_feader import QueueFeader
from task.metrics import stance_f1
from log import log
import path
import os
import pickle
import threading
from models.baselines import svm
from models.transformer.tranformer_nli import transformer_nli
from models.transformer.transformer_controversy import transformer_controversy, transformer_controversy_fixed_encoding
from models.transformer.transformer_adhoc import transformer_adhoc
from models.transformer.transformer_lm import transformer_ql

from tensorflow.python.client import timeline
from misc_lib import delete_if_exist
from attribution.deleter_trsfmr import *
from attribution.baselines import *
from evaluation import *
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
        self.log2 = log.aux_logger()
        self.model_dir = path.model_path
        self.sess = None
        self.g_step = 0
    @staticmethod
    def init_sess():
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False
                                )
        config.gpu_options.allow_growth = True

        return tf.Session(config=config)

    def get_train_op(self, loss, name='Adam'):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.hparam.lr, beta1=0.9, beta2=0.98, epsilon=1e-8,
                                           name=name)
        train_op = optimizer.minimize(loss, global_step=self.global_step)
        return train_op


    def get_train_op_with_score(self, loss, scope, name='Adam'):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        tvars = tf.trainable_variables()
        g_vars = ([var for var in tvars if var.name.split("/")[0] in scope])
        print("Trainable variables")
        for v in g_vars:
            print(v)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.hparam.lr, beta1=0.9, beta2=0.98, epsilon=1e-8,
                                           name=name)
        train_op = optimizer.minimize(loss, global_step=self.global_step, var_list=g_vars)
        return train_op


    def get_train_op_with_black_list(self, loss, exclude_scope, name='Adam'):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        tvars = tf.trainable_variables()
        g_vars = ([var for var in tvars if not var.name.startswith(exclude_scope)])
        print("Trainable variables")
        for v in g_vars:
            print(v)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.hparam.lr, beta1=0.9, beta2=0.98, epsilon=1e-8,
                                           name=name)
        train_op = optimizer.minimize(loss, global_step=self.global_step, var_list=g_vars)
        return train_op

    def temp_saver(self):
        self.save_model("interval")


    def stance_baseline(self, topic, voca_path):
        print("Experiment.stance_baseline()")
        max_sequence= 140
        stance_data = stance_detection.DataLoader(topic, max_sequence, voca_path)

        stance_data.load_train_data()
        stance_data.load_test_data()
        label_count = Counter()

        for entry in stance_data.train_data_raw:
            label_count[entry["label"]] += 1

        common_label, _ = label_count.most_common(1)[0]
        pred_major = np.array([common_label] * len(stance_data.dev_data_raw))


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


    def feature_svm(self, voca_size, stance_data, preload_id):
        print("Experiment.feature_svm()")
        feature_loc = 0
        task = TransformerClassifier(self.hparam, voca_size, stance_detection.num_classes, True, feature_loc)


        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        name = preload_id[0]
        id = preload_id[1]
        self.load_model_encoder(name, id)
        train_batches = get_batches(stance_data.get_train_data(), self.hparam.batch_size)
        dev_batches = get_batches(stance_data.get_test_data(), self.hparam.batch_size)

        def feature_combine(features):
            return np.average(features, axis=1)
            #n_batch = features.shape[0]
            #return np.reshape(features, [n_batch,-1])

        def convert2feature(batches):
            X = []
            Y = []
            for batch in batches:
                input, target = batch
                enc, = self.sess.run([task.enc, ],
                                    feed_dict={
                                        task.x: input,
                                        task.y: target,
                                    })
                X.append(enc)
                Y.append(target)
                # enc : [batch, seq, dim]
            X = feature_combine(np.concatenate(X, axis=0))
            Y = np.concatenate(Y, axis=0)
            return X, Y

        train_X, train_y = convert2feature(train_batches)
        dev_X, dev_y = convert2feature(dev_batches)
        scaler = StandardScaler()
        scaler.fit(train_X)
        train_X = scaler.transform(train_X)
        dev_X = scaler.transform(dev_X)

        for param_c in [1e-4, 1e-3, 1e-2,1e-1]:
            svclassifier = LinearSVC(C=param_c)
            svclassifier.fit(train_X, train_y)
            train_pred = svclassifier.predict(train_X)
            dev_pred = svclassifier.predict(dev_X)
            train_f1 = stance_f1(train_pred, train_y)
            dev_f1 = stance_f1(dev_pred, dev_y)
            print("Train : {0:.2f}  Dev : {1:.2f}".format(train_f1, dev_f1))

        return

    def train_stance(self, voca_size, stance_data, preload_id = None):
        print("Experiment.train_stance()")
        valid_freq = 10
        f_finetune = (preload_id is not None)
        if f_finetune:
            #feature_loc = int(self.hparam.seq_max / 2)
            feature_loc = self.hparam.sent_max
            feature_loc = 0
            print("feature_loc", feature_loc)
        else:
            feature_loc = 0
        task = TransformerClassifier(self.hparam, voca_size, stance_detection.num_classes, True, feature_loc)
        num_param = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print("Total parameters : {}".format(num_param))
        def get_train_op_only_top(loss):
            def fine_tune(v):
                print(v)
                tokens = v.name.split('/')
                if tokens[0] == 'cls_dense':
                    return True
                if tokens[0] == "encoder" and tokens[1] == "num_blocks_11":
                    return True
                return False

            target = list(filter(fine_tune, tf.trainable_variables()))
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.hparam.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
            train_op = optimizer.minimize(loss, global_step=self.global_step, var_list=target)
            return train_op

        #train_op = get_train_op_only_top(task.loss)
        train_op = self.get_train_op(task.loss)

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())

        if preload_id is not None:
            name = preload_id[0]
            id = preload_id[1]
            self.load_model_encoder(name, id)
        random.seed(0)

        train_batches = get_batches(stance_data.get_train_data(), self.hparam.batch_size)
        dev_batches = get_batches(stance_data.get_test_data(), self.hparam.batch_size)

        def train_fn(batch, step_i):
            loss_val = batch_train(self.sess, batch, train_op, task)
            self.log.debug("[Train] Step {0} loss={1:.04f}".format(step_i, loss_val))
            return loss_val, 0

        valid_history = []
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
            avg_loss, avg_acc = average(loss_list), average(acc_list)
            self.log.info("[Dev] F1={0:.02f} Acc={1:.02f} loss={2:.04f}".format(f1, avg_acc, avg_loss))
            valid_history.append((avg_acc, f1))
            return

        num_epoch = self.hparam.num_epochs
        print("Start Training")
        for i in range(num_epoch):
            loss, _ = epoch_runner(train_batches, train_fn,
                                   valid_fn, valid_freq,
                                   self.temp_saver, self.save_interval)
            self.log.info("[Train] Epoch {} Done. Loss={}".format(i, loss))

        valid_fn()
        self.save_model("after_stance")
        return valid_history


    def train_aux1(self, hp2, voca_size, aux_data, preload_id = None):
        print("Experiment.train_aux1()")
        valid_freq = 10
        task = AuxLM(self.hparam, hp2, voca_size, True)

        def get_train_op_only_top(loss):
            def fine_tune(v):
                print(v)
                tokens = v.name.split('/')
                if tokens[0] == 'cls_dense':
                    return True
                if tokens[0] == "encoder" and tokens[1] == "num_blocks_11":
                    return True
                return False

            target = list(filter(fine_tune, tf.trainable_variables()))
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.hparam.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
            train_op = optimizer.minimize(loss, global_step=self.global_step, var_list=target)
            return train_op

        train_aux = self.get_train_op(task.aux_loss)

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())

        if preload_id is not None:
            name = preload_id[0]
            id = preload_id[1]
            self.load_model_encoder(name, id)
        random.seed(0)

        train_batches = get_batches(aux_data.get_train_data(), self.hparam.batch_size)
        dev_batches = get_batches(aux_data.get_test_data(), self.hparam.batch_size)

        def train_fn(batch, step_i):
            input, target = batch
            loss_val, _ = self.sess.run([task.aux_loss, train_aux],
                                   feed_dict={
                                       task.x: input,
                                       task.y_aux: target,
                                   },
                                   )
            self.log.debug("[Train] Step {0} loss={1:.04f}".format(step_i, loss_val))
            return loss_val, 0

        valid_history = []
        def valid_fn():
            loss_list = []
            acc_list = []
            gold_list = []
            for batch in dev_batches:
                input, target = batch
                loss_val, acc, = self.sess.run([task.aux_loss, task.aux_acc, ],
                                    feed_dict={
                                        task.x: input,
                                        task.y_aux: target,
                                    })
                loss_list.append(loss_val)
                acc_list.append(acc)
                gold_list.append(target)

            avg_loss, avg_acc = average(loss_list), average(acc_list)
            self.log.info("[Dev] Acc={0:.02f} loss={1:.04f}".format(avg_acc, avg_loss))
            valid_history.append((avg_acc, f1))
            return

        num_epoch = self.hparam.num_epochs
        print("Start Training")
        for i in range(3):
            loss, _ = epoch_runner(train_batches, train_fn,
                                   valid_fn, valid_freq,
                                   self.temp_saver, self.save_interval)
            self.log.info("[Train] Epoch {} Done. Loss={}".format(i, loss))

        valid_fn()
        self.save_model("after_aux")
        return valid_history

    def train_aux2(self, hp2, exp_config, data_generator, preload_id = None):
        print("Experiment.train_aux2()")
        valid_freq = 10
        task = AuxLM(self.hparam, hp2, data_generator.voca_size, True)

        def get_train_non_aux(loss):
            def fine_tune(v):
                tokens = v.name.split('/')
                if tokens[0] == 'aux':
                    return False
                return True

            target = list(filter(fine_tune, tf.trainable_variables()))
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.hparam.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
            train_op = optimizer.minimize(loss, global_step=self.global_step, var_list=target)
            return train_op

        train_lm = get_train_non_aux(task.loss)

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)

        def load_model(name, id):
            run_dir = os.path.join(self.model_dir, 'runs')
            if "reserve/" in name:
                tokens = name.split("/")
                run_dir = os.path.join(run_dir, tokens[0])
                name = tokens[1]
            save_dir = os.path.join(run_dir, name)
            path = os.path.join(save_dir, "model-{}".format(id))
            variables = tf.contrib.slim.get_variables_to_restore()
            def condition(v):
                if 'Adam' in v.name:
                    return False
                return True

            variables_to_restore = [v for v in variables if condition(v)]
            print(variables_to_restore)
            print("Restoring: {} {}".format(name, id))

            loader = tf.train.Saver(variables_to_restore, max_to_keep=1)
            loader.restore(self.sess, path)

        name = preload_id[0]
        id = preload_id[1]
        load_model(name, id)
        random.seed(0)

        def generate_train_batch():
            batch_size = self.hparam.batch_size
            data = data_generator.get_train_instances(batch_size)
            batches = get_batches_ex(list(data), batch_size, 2)
            return batches[0]

        def get_dev_batches():
            num_batches = 10
            test_data = data_generator.get_test_instances(num_batches * self.hparam.batch_size)
            test_batches = get_batches_ex(list(test_data), self.hparam.batch_size, 2)
            return test_batches

        print("Generate dev batches")
        dev_batch = list(get_dev_batches())

        print("Init queue feader ")
        max_reserve_batch = 100
        queue_feader = QueueFeader(max_reserve_batch, generate_train_batch)

        def batch2feed_dict(batch):
            x, y  = batch
            feed_dict = {
                task.x: x,
                task.y: y,
            }
            return feed_dict


        def train_fn(batch, step_i):
            loss_val, summary, _ = self.sess.run([task.loss, self.merged, train_lm,
                                                  ],
                                                 feed_dict=batch2feed_dict(batch)
                                                 )
            self.log.debug("Step {0} train loss={1:.04f}".format(step_i, loss_val))
            self.train_writer.add_summary(summary, self.g_step)
            self.g_step += 1
            return loss_val, 0

        valid_history = []

        def valid_fn():
            loss_list = []
            for batch in dev_batch:
                loss_val, summary = self.sess.run([task.loss, self.merged],
                                                  feed_dict=batch2feed_dict(batch)
                                                  )
                loss_list.append(loss_val)

                self.test_writer.add_summary(summary, self.g_step)
            self.log.info("Validation : loss={0:.04f}".format(average(loss_list)))

        def save_fn():
            self.save_model(exp_config.name)

        print("Start Training")
        last_save = time.time()
        max_step = 1000 * 1000 * 1000
        for step_i in range(max_step):
            if step_i % valid_freq == 0:
                valid_fn()

            if save_fn is not None:
                if time.time() - last_save > self.save_interval:
                    save_fn()
                    last_save = time.time()

            batch = queue_feader.get()
            train_fn(batch, step_i)
            self.g_step += 1


    def train_aux_stance(self, hp2, voca_size, stance_data, preload_id):
        print("Experiment.train_stance()")
        valid_freq = 10
        task = AuxClassification(self.hparam, hp2, voca_size, stance_detection.num_classes, True)

        def get_train_non_aux(loss):
            def fine_tune(v):
                tokens = v.name.split('/')
                #if tokens[0] == 'aux':
                #    return False
                return True

            target = list(filter(fine_tune, tf.trainable_variables()))
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.hparam.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
            train_op = optimizer.minimize(loss, global_step=self.global_step, var_list=target)
            return train_op

        train_op = get_train_non_aux(task.loss)

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        def load_model(name, id):
            run_dir = os.path.join(self.model_dir, 'runs')
            save_dir = os.path.join(run_dir, name)
            path = os.path.join(save_dir, "model-{}".format(id))
            variables = tf.contrib.slim.get_variables_to_restore()
            def condition(v):
                if 'Adam' in v.name:
                    return False
                if 'aux' in v.name:
                    return True
                return False

            variables_to_restore = [v for v in variables if condition(v)]
            print(variables_to_restore)
            print("Restoring: {} {}".format(name, id))

            loader = tf.train.Saver(variables_to_restore, max_to_keep=1)
            loader.restore(self.sess, path)

        if preload_id is not None:
            name = preload_id[0]
            id = preload_id[1]
            load_model(name, id)
        random.seed(0)

        train_batches = get_batches(stance_data.get_train_data(), self.hparam.batch_size)
        dev_batches = get_batches(stance_data.get_test_data(), self.hparam.batch_size)

        def train_fn(batch, step_i):
            loss_val = batch_train(self.sess, batch, train_op, task)
            self.log.debug("[Train] Step {0} loss={1:.04f}".format(step_i, loss_val))
            return loss_val, 0

        valid_history = []
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
            avg_loss, avg_acc = average(loss_list), average(acc_list)
            self.log.info("[Dev] F1={0:.02f} Acc={1:.02f} loss={2:.04f}".format(f1, avg_acc, avg_loss))
            valid_history.append((avg_acc, f1))
            return

        num_epoch = self.hparam.num_epochs
        print("Start Training")
        for i in range(num_epoch):
            loss, _ = epoch_runner(train_batches, train_fn,
                                   valid_fn, valid_freq,
                                   self.temp_saver, self.save_interval)
            self.log.info("[Train] Epoch {} Done. Loss={}".format(i, loss))

        valid_fn()
        self.save_model("after_stance")
        return valid_history


    def train_stance_consistency(self, voca_size, stance_data, aux_data):
        print("Experiment.train_stance_consistency()")
        valid_freq = 10
        feature_loc = 0
        task = ConsistentClassifier(self.hparam, voca_size, stance_detection.num_classes, True, feature_loc)

        train_op = self.get_train_op(task.loss)
        train_aux = self.get_train_op(task.consist_loss)

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())

        random.seed(0)

        train_batches = get_batches(stance_data.get_train_data(), self.hparam.batch_size_sup)
        dev_batches = get_batches(stance_data.get_dev_data(), 16)
        max_reserve_batch = 10
        n_step = len(train_batches)

        def generate_aux_epoch():
            batch_size = self.hparam.batch_size_aux
            data = aux_data.get_insts(batch_size * n_step)
            batches = get_batches_ex(list(data), batch_size, 1)
            assert len(batches) == n_step
            assert batches[0][0].shape == (batch_size, 2, self.hparam.seq_max)
            return batches

        queue_feader = QueueFeader(max_reserve_batch, generate_aux_epoch)


        def train_fn(batch, step_i):
            s_batch, aux_input = batch
            input, target = s_batch
            x_pair, = aux_input
            loss_val, s_loss, c_loss, idk_loss, \
            pred_pair, \
            _ = self.sess.run([task.loss, task.supervised_loss, task.consist_loss, task.idk_loss,
                               task.pred_pair,
                               train_op],
                                   feed_dict={
                                       task.x: input,
                                       task.y: target,
                                       task.x_pair: x_pair
                                   },
                                   )
            self.log.info("[Train] Step {0} loss={1:.04f} s_loss={2:.04f} c_loss={3:.04f} idk_loss={4:.04f}"
                           .format(step_i, loss_val, s_loss, c_loss, idk_loss))

            pred_label =np.argmax(pred_pair, axis=-1)
            non_zero = np.count_nonzero(pred_label)

            self.log.debug("non_zero={0} ".format(non_zero))
            self.log.debug("pred_label={0} ".format(pred_label))
            return loss_val, 0


        def train_aux_fn(batch, step_i):
            x_pair, = batch
            c_loss, _ = self.sess.run([task.consist_loss, train_aux],
                                   feed_dict={
                                       task.x_pair: x_pair
                                   },
                                   )
            self.log.debug("[Train Aux] Step {0} c_loss={1:.04f}"
                           .format(step_i, c_loss))
            return c_loss, 0


        valid_history = []
        def valid_fn():
            loss_list = []
            acc_list = []
            logits_list = []
            gold_list = []
            for batch in dev_batches:
                input, target = batch
                loss_val, acc, logits = self.sess.run([task.supervised_loss, task.acc, task.logits],
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
            avg_loss, avg_acc = average(loss_list), average(acc_list)
            self.log.info("[Dev] F1={0:.02f} Acc={1:.02f} s_loss={2:.04f}".format(f1, avg_acc, avg_loss))
            valid_history.append((avg_acc, f1))
            return

        num_epoch = self.hparam.num_epochs
        print("Start Training")
        for i in range(num_epoch):
            aux_baches_per_epoch = queue_feader.get()
            t_batches = list(zip(train_batches, aux_baches_per_epoch))
            loss, _ = epoch_runner(t_batches, train_fn,
                                   valid_fn, valid_freq,
                                   self.temp_saver, self.save_interval)
            self.log.info("[Train] Epoch {} Done. Loss={}".format(i, loss))


        valid_fn()
        return valid_history


    def train_stance_pair_feature(self, voca_size, stance_data, preload_id = None):
        print("Experiment.train_stance_pair_feature()")
        valid_freq = 10
        f_finetune = (preload_id is not None)

        task = PairFeatureClassification(self.hparam, voca_size, stance_detection.num_classes, True)

        train_op = self.get_train_op(task.loss)

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())

        def load_model(name, id):
            run_dir = os.path.join(self.model_dir, 'runs')
            save_dir = os.path.join(run_dir, name)
            path = os.path.join(save_dir, "model-{}".format(id))

            variables = tf.contrib.slim.get_variables_to_restore()

            def condition(v):
                if v.name.split('/')[0] == 'feature_encoder':
                    return True
                return False

            variables_to_restore = [v for v in variables if condition(v)]
            print("Restoring:")
            for v in variables_to_restore:
                print(v)

            self.loader = tf.train.Saver(variables_to_restore, max_to_keep=1)
            self.loader.restore(self.sess, path)

        if preload_id is not None:
            name = preload_id[0]
            id = preload_id[1]
            load_model(name, id)
        random.seed(0)

        train_batches = get_batches(stance_data.get_train_data(), self.hparam.batch_size)
        dev_batches = get_batches(stance_data.get_test_data(), self.hparam.batch_size)

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
                loss_val, acc, logits, feature, = self.sess.run([task.loss, task.acc, task.logits, task.feature1],
                                    feed_dict={
                                        task.x: input,
                                        task.y: target,
                                    })
                print(feature[0])
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

        num_epoch = self.hparam.num_epochs
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
            self.log.debug("Step {0} train loss={1:.04f}".format(step_i,
                                                                 loss_val))
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

    def setup_summary_writer(self, exp_name):
        summary_path = os.path.join(path.output_path, "summary")
        exist_or_mkdir(summary_path)
        summary_run_path = os.path.join(summary_path, exp_name)
        exist_or_mkdir(summary_run_path)

        self.run_metadata = tf.RunMetadata()
        train_log_path = os.path.join(summary_run_path, "train")
        test_log_path = os.path.join(summary_run_path, "test")
        delete_if_exist(train_log_path)
        delete_if_exist(test_log_path)
        self.train_writer = tf.summary.FileWriter(train_log_path,
                                                  self.sess.graph)
        self.test_writer = tf.summary.FileWriter(test_log_path,
                                                 self.sess.graph)

    def clear_run(self):
        tf.reset_default_graph()

    def train_pair_lm(self, exp_config, data):
        print("train_pair_lm")
        valid_freq = exp_config.valid_freq
        save_interval = exp_config.save_interval

        task = TransformerPairLM(self.hparam, data.voca_size, 2, True)
        train_op = self.get_train_op(task.loss)
        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)

        def prepare_data():
            n_inputs = 3

            train_data = data.get_train_batch()
            train_batches = get_batches_ex(list(train_data), self.hparam.batch_size, n_inputs)

            test_data = data.get_test_generator()
            test_batches = get_batches_ex(list(test_data), self.hparam.batch_size, n_inputs)
            return train_batches, test_batches

        use_cache = False
        if not use_cache:
            print("Generating data")
            train_batches, test_batches = prepare_data()
            pickle.dump((train_batches, test_batches), open("batch_cache.pickle", "wb"))
        else:
            train_batches, test_batches = pickle.load(open("batch_cache.pickle", "rb"))

        x, ys, yc = train_batches[0]
        print("Batch per Epoch : {}".format(len(train_batches)))
        print(x[0])
        print(ys[0])
        print(yc[0])
        def batch2feed_dict(batch):
            x, y_seq, y_cls = batch
            feed_dict = {
                task.x: x,
                task.y_seq: y_seq,
                task.y_cls: y_cls,
            }
            return feed_dict

        def train_fn(batch, step_i):
            loss_val, summary, _ ,\
                cls_pred, seq_logits = self.sess.run([task.loss, self.merged, train_op,
                                                  task.preds, task.seq_logits,
                                                  ],
                                        feed_dict = batch2feed_dict(batch)
                                        )
            self.log.debug("Step {0} train loss={1:.04f}".format(step_i, loss_val))
            self.train_writer.add_summary(summary, self.g_step)
            self.g_step += 1
            return loss_val, 0

        def valid_fn():
            loss_list = []
            for batch in test_batches[:20]:
                loss_val, summary = self.sess.run([task.loss, self.merged],
                                          feed_dict = batch2feed_dict(batch)
                                          )
                loss_list.append(loss_val)

                self.test_writer.add_summary(summary, self.g_step)
            self.log.info("Validation : loss={0:.04f}".format(average(loss_list)))

        def save_fn():
            self.save_model(exp_config.name)

        for epoch_i in range(exp_config.num_epoch):
            epoch_runner(train_batches, train_fn, valid_fn, valid_freq, save_fn, save_interval, False)

        self.save_model(exp_config.name+"_final")


    def train_pair_lm_inf(self, exp_config, data_generator):
        print("train_pair_lm_inf")
        valid_freq = exp_config.valid_freq
        max_step = 1000 * 1000 * 1000
        save_interval = exp_config.save_interval

        task = TransformerPairLM(self.hparam, data_generator.voca_size, 2, True)
        train_op = self.get_train_op(task.loss)
        n_inputs = 3
        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)
        self.g_step = 0


        def get_dev_batches():
            num_batches = 10
            test_data = data_generator.get_test_generator(num_batches * self.hparam.batch_size)
            test_batches = get_batches_ex(list(test_data), self.hparam.batch_size, n_inputs)
            return test_batches

        print("Generate dev batches")
        dev_batch = list(get_dev_batches())

        def generate_train_batch():
            batch_size = self.hparam.batch_size
            data = data_generator.get_train_batch(batch_size)
            batches = get_batches_ex(list(data), batch_size, n_inputs)
            return batches[0]

        print("Init queue feader ")
        max_reserve_batch = 100
        queue_feader = QueueFeader(max_reserve_batch, generate_train_batch)

        def batch2feed_dict(batch):
            x, y_seq, y_cls = batch
            feed_dict = {
                task.x: x,
                task.y_seq: y_seq,
                task.y_cls: y_cls,
            }
            return feed_dict

        def train_fn(batch, step_i):
            loss_val, summary, _ , \
            cls_pred, seq_logits = self.sess.run([task.loss, self.merged, train_op,
                                                  task.preds, task.seq_logits,
                                                  ],
                                                 feed_dict = batch2feed_dict(batch)
                                                 )
            self.log.debug("Step {0} train loss={1:.04f}".format(step_i, loss_val))
            self.train_writer.add_summary(summary, self.g_step)
            self.g_step += 1
            return loss_val, 0

        def valid_fn():
            loss_list = []
            for batch in dev_batch:
                loss_val, summary = self.sess.run([task.loss, self.merged],
                                                  feed_dict = batch2feed_dict(batch)
                                                  )
                loss_list.append(loss_val)

                self.test_writer.add_summary(summary, self.g_step)
            self.log.info("Validation : loss={0:.04f}".format(average(loss_list)))



        def save_fn():
            self.save_model(exp_config.name)

        last_save = time.time()
        for step_i in range(max_step):
            if step_i % valid_freq == 0:
                valid_fn()

            if save_fn is not None:
                if time.time() - last_save > save_interval:
                    save_fn()
                    last_save = time.time()

            batch = queue_feader.get()

            x, ys, yc = batch
            train_fn(batch, step_i)
            self.g_step += 1


    def train_doc_lm(self, exp_config, data_generator):
        print("train_doc_lm")
        valid_freq = exp_config.valid_freq
        max_step = 1000 * 1000 * 1000
        save_interval = exp_config.save_interval

        task = TransformerLM(self.hparam, data_generator.voca_size, True)
        train_op = self.get_train_op(task.loss)
        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)

        def get_dev_batches():
            num_batches = 10
            test_data = data_generator.get_test_instances(num_batches * self.hparam.batch_size)
            test_batches = get_batches_ex(list(test_data), self.hparam.batch_size, 2)
            return test_batches

        print("Generate dev batches")
        dev_batch = list(get_dev_batches())

        def generate_train_batch():
            batch_size = self.hparam.batch_size
            data = data_generator.get_train_instances(batch_size)
            batches = get_batches_ex(list(data), batch_size, 2)
            return batches[0]

        print("Init queue feader ")
        max_reserve_batch = 100
        queue_feader = QueueFeader(max_reserve_batch, generate_train_batch)

        def batch2feed_dict(batch):
            x, y  = batch
            feed_dict = {
                task.x: x,
                task.y: y,
            }
            return feed_dict

        def train_fn(batch, step_i):
            loss_val, summary, _ = self.sess.run([task.loss, self.merged, train_op,
                                                  ],
                                                 feed_dict=batch2feed_dict(batch)
                                                 )
            self.log.debug("Step {0} train loss={1:.04f}".format(step_i, loss_val))
            self.train_writer.add_summary(summary, self.g_step)
            self.g_step += 1
            return loss_val, 0

        def valid_fn():
            loss_list = []
            for batch in dev_batch:
                loss_val, summary = self.sess.run([task.loss, self.merged],
                                                  feed_dict=batch2feed_dict(batch)
                                                  )
                loss_list.append(loss_val)

                self.test_writer.add_summary(summary, self.g_step)
            self.log.info("Validation : loss={0:.04f}".format(average(loss_list)))

        def save_fn():
            self.save_model(exp_config.name, 100)

        last_save = time.time()
        for step_i in range(max_step):
            if step_i % valid_freq == 0:
                valid_fn()

            if save_fn is not None:
                if time.time() - last_save > save_interval:
                    save_fn()
                    last_save = time.time()

            batch = queue_feader.get()

            x, y  = batch
            train_fn(batch, step_i)
            self.g_step += 1

    def train_pair_feature(self, exp_config, data_generator):
        print("train_pair_feature")
        valid_freq = exp_config.valid_freq
        max_step = 1000 * 100
        save_interval = exp_config.save_interval

        task = PairFeature(self.hparam, data_generator.voca_size, True)
        train_op = self.get_train_op(task.loss)
        n_inputs = 3
        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)


        def get_dev_batches():
            num_batches = 10
            test_data = data_generator.get_test_generator(num_batches * self.hparam.batch_size)
            test_batches = get_batches_ex(list(test_data), self.hparam.batch_size, n_inputs)
            return test_batches

        print("Generate dev batches")
        dev_batch = list(get_dev_batches())

        def generate_train_batch():
            batch_size = self.hparam.batch_size
            data = data_generator.get_train_batch(batch_size)
            batches = get_batches_ex(list(data), batch_size, n_inputs)
            return batches[0]

        print("Init queue feader ")
        max_reserve_batch = 100
        queue_feader = QueueFeader(max_reserve_batch, generate_train_batch)

        def batch2feed_dict(batch):
            x, y_seq, y_cls = batch
            feed_dict = {
                task.x: x,
                task.y_cls: y_cls,
            }
            return feed_dict

        def train_fn(batch, step_i):
            loss_val, summary, _ , = self.sess.run([task.loss, self.merged, train_op,
                                                  ],
                                                 feed_dict = batch2feed_dict(batch)
                                                 )
            self.log.debug("Step {0} train loss={1:.04f}".format(step_i, loss_val))
            self.train_writer.add_summary(summary, self.g_step)
            self.g_step += 1
            return loss_val, 0

        def valid_fn():
            loss_list = []
            for batch in dev_batch:
                loss_val, summary = self.sess.run([task.loss, self.merged],
                                                  feed_dict = batch2feed_dict(batch)
                                                  )
                loss_list.append(loss_val)

                self.test_writer.add_summary(summary, self.g_step)
            self.log.info("Validation : loss={0:.04f}".format(average(loss_list)))



        def save_fn():
            self.save_model(exp_config.name, 30)

        last_save = time.time()
        for step_i in range(max_step):
            if step_i % valid_freq == 0:
                valid_fn()

            if save_fn is not None:
                if time.time() - last_save > save_interval:
                    save_fn()
                    last_save = time.time()

            batch = queue_feader.get()

            train_fn(batch, step_i)
            self.g_step += 1

    def save_model(self, name, max_to_keep = 1):
        run_dir = os.path.join(self.model_dir, 'runs')
        save_dir = os.path.join(run_dir, name)

        exist_or_mkdir(self.model_dir)
        exist_or_mkdir(run_dir)
        exist_or_mkdir(save_dir)

        path = os.path.join(save_dir, "model")
        if self.saver is None:
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=max_to_keep)
        ret = self.saver.save(self.sess, path, global_step=self.global_step)
        self.log.info("Model saved at {} - {}".format(path, ret))


    def load_model_encoder(self, name, id):
        run_dir = os.path.join(self.model_dir, 'runs')
        if "reserve/" in name:
            tokens = name.split("/")
            run_dir = os.path.join(run_dir, tokens[0])
            name = tokens[1]
        save_dir = os.path.join(run_dir, name)
        path = os.path.join(save_dir, "model-{}".format(id))

        variables = tf.contrib.slim.get_variables_to_restore()

        def condition(v):
            return v.name.split('/')[0] == 'encoder'

        variables_to_restore = [v for v in variables if condition(v)]
        print("Restoring: {} {}".format(name, id))

        self.loader = tf.train.Saver(variables_to_restore, max_to_keep=1)
        self.loader.restore(self.sess, path)

    def load_model_b1(self, name, id, exclude_namespace):
        run_dir = os.path.join(self.model_dir, 'runs')
        if "reserve/" in name:
            tokens = name.split("/")
            run_dir = os.path.join(run_dir, tokens[0])
            name = tokens[1]
        save_dir = os.path.join(run_dir, name)
        path = os.path.join(save_dir, "model-{}".format(id))

        def condition(v):
            if v.name.split('/')[0] in exclude_namespace:
                return False
            if v.name.split('/')[-1] in exclude_namespace:
                return False
            return True


        variables = tf.contrib.slim.get_variables_to_restore()
        variables_to_restore = [v for v in variables if condition(v)]

        print("Restoring: {} {}".format(name, id))

        self.loader = tf.train.Saver(variables_to_restore, max_to_keep=1)
        self.loader.restore(self.sess, path)

    def load_model_white(self, name, id, include_namespace):
        run_dir = os.path.join(self.model_dir, 'runs')
        save_dir = os.path.join(run_dir, name)
        path = os.path.join(save_dir, "{}".format(id))

        def condition(v):
            if v.name.split('/')[0] in include_namespace:
                return True
            return False

        variables = tf.contrib.slim.get_variables_to_restore()
        variables_to_restore = [v for v in variables if condition(v)]
        print("Restoring: {} {}".format(name, id))
        for v in variables_to_restore:
            print(v)

        self.loader = tf.train.Saver(variables_to_restore, max_to_keep=1)
        self.loader.restore(self.sess, path)


    def load_model_bert(self, name, id):
        run_dir = os.path.join(self.model_dir, 'runs')

        save_dir = os.path.join(run_dir, name)
        path = os.path.join(save_dir, "{}".format(id))

        def condition(v):
            if v.name.split('/')[0] in ['bert']:
                return True
            return False

        variables = tf.contrib.slim.get_variables_to_restore()
        variables_to_restore = [v for v in variables if condition(v)]
        for v in variables_to_restore:
            print(v)

        print("Restoring: {} {}".format(name, id))
        self.loader = tf.train.Saver(variables_to_restore, max_to_keep=1)
        self.loader.restore(self.sess, path)

    def load_model_all(self, name, id):
        run_dir = os.path.join(self.model_dir, 'runs')
        save_dir = os.path.join(run_dir, name)
        path = os.path.join(save_dir, "{}".format(id))

        variables = tf.contrib.slim.get_variables_to_restore()
        print("Restoring: {} {}".format(name, id))
        self.loader = tf.train.Saver(variables, max_to_keep=1)
        self.loader.restore(self.sess, path)


    def load_nli_data(self, data_loader):
        # load data
        print("Loading Data")
        pickle_name = "nli_{}".format(self.hparam.batch_size)
        pickle_path = os.path.join(path.cache_path, pickle_name)

        use_pickle = True
        if use_pickle :
            train_batches, dev_batches = pickle.load(open(pickle_path, "rb"))
        else:
            train_batches = get_batches_ex(data_loader.get_train_data(), self.hparam.batch_size, 4)
            dev_batches = get_batches_ex(data_loader.get_dev_data(), self.hparam.batch_size, 4)
            pickle.dump((train_batches, dev_batches), open(pickle_path, "wb"))
        return train_batches, dev_batches

    def train_nli(self, nli_setting, exp_config, data_loader):
        print("train_nli")
        task = transformer_nli(self.hparam, nli_setting.vocab_size)
        train_op = self.get_train_op(task.loss)

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)

        def batch2feed_dict(batch):
            x0,x1,x2, y  = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y: y,
            }
            return feed_dict

        def train_fn(batch, step_i):
            loss_val, summary, _ = self.sess.run([task.loss, self.merged, train_op,
                                             ],
                                            feed_dict=batch2feed_dict(batch)
                                            )
            self.log.debug("Step {0} train loss={1:.04f}".format(step_i, loss_val))
            self.train_writer.add_summary(summary, self.g_step)
            self.g_step += 1
            return loss_val, 0

        train_batches, dev_batches = self.load_nli_data(data_loader)

        def valid_fn():
            loss_list = []
            for batch in dev_batches[:100]:
                loss_val, summary = self.sess.run([task.loss, self.merged],
                                                  feed_dict=batch2feed_dict(batch)
                                                  )
                loss_list.append(loss_val)

                self.test_writer.add_summary(summary, self.g_step)
            self.log.info("Validation : loss={0:.04f}".format(average(loss_list)))

        valid_freq = 100
        # train_op
        print("Train epoch")
        num_epochs = exp_config.num_epoch
        for i_epoch in range(num_epochs):
            loss, _ = epoch_runner(train_batches, train_fn,
                                   valid_fn, valid_freq,
                                   self.temp_saver, self.save_interval)

    def nli_explain_baselines(self, nli_setting, exp_config, data_loader, preload_id):
        enc_explain_dev, explain_dev = data_loader.get_dev_explain()
        train_batches, dev_batches = self.load_nli_data(data_loader)

        task = transformer_nli(self.hparam, nli_setting.vocab_size)
        CONTRADICTION = 2

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        if preload_id is not None:
            name = preload_id[0]
            id = preload_id[1]
            if exp_config.load_names :
                self.load_model_white(name, id, exp_config.load_names)
            elif name in exp_config.name:
                self.load_model_all(name, id)
            elif "NLI" in name:
                self.load_model_white(name, id, ['bert', 'dense_cls'])
            else:
                self.load_model_bert(name, id)


        def forward_run(inputs):
            batches = get_batches_ex(inputs, self.hparam.batch_size, 3)
            logit_list = []
            for batch in batches:
                x0, x1, x2 = batch
                soft_out,  = self.sess.run([task.sout, ],
                                               feed_dict={
                                                task.x_list[0]: x0,
                                                task.x_list[1]: x1,
                                                task.x_list[2]: x2,
                                               })
                logit_list.append(soft_out)
            return np.concatenate(logit_list)



        idf_scorer = IdfScorer(train_batches)


        def eval(method):
            print("Eval")
            begin = time.time()
            explains = method(enc_explain_dev, 'conflict', forward_run)
            assert len(explains) == len(explain_dev)
            end = time.time()
            print("Elapsed Time : {}".format(end- begin))

            p_at_1, MAP_score = eval_explain(explains, data_loader)
            print("P@1\t{}".format(p_at_1))
            print("MAP\t{}".format(MAP_score))

        todo_list = [
                     ('random', explain_by_random),
                    ('idf', idf_scorer.explain),
                    ('deletion', explain_by_deletion),
                     ]
        for method_name, method in todo_list:
            print(method_name)
            eval(method)



    def train_nli_ex(self, nli_setting, exp_config, data_loader, preload_id, f_train_ex):
        enc_explain_dev, explain_dev = data_loader.get_dev_explain()

        task = transformer_nli(self.hparam, nli_setting.vocab_size)
        with tf.variable_scope("optimizer"):
            train_cls = self.get_train_op(task.loss)
            train_rl = self.get_train_op(task.rl_loss, name="rl")

        CONTRADICTION = 2

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)

        if preload_id is not None:
            name = preload_id[0]
            id = preload_id[1]
            if exp_config.load_names :
                self.load_model_white(name, id, exp_config.load_names)
            elif name in exp_config.name:
                self.load_model_all(name, id)
            elif "NLI" in name:
                self.load_model_white(name, id, ['bert', 'dense_cls'])
            else:
                self.load_model_bert(name, id)


        def batch2feed_dict(batch):
            x0, x1, x2, y  = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y: y,
            }
            return feed_dict

        def eval():
            print("Eval")
            begin = time.time()
            batches = get_batches_ex(enc_explain_dev, self.hparam.batch_size, 3)

            conf_logit_list = []
            for batch in batches:
                x0, x1, x2 = batch
                logits, conf_logit = self.sess.run([task.sout, task.conf_logits ],
                                               feed_dict={
                                                task.x_list[0]: x0,
                                                task.x_list[1]: x1,
                                                task.x_list[2]: x2,
                                               })
                conf_logit_list.append(conf_logit)
            conf_logit = np.concatenate(conf_logit_list)
            assert len(conf_logit) == len(explain_dev)
            end = time.time()
            print("Elapsed Time : {}".format(end- begin))

            p_at_1, MAP_score = eval_explain(conf_logit, data_loader)
            print("P@1 : {}".format(p_at_1))
            print("MAP : {}".format(MAP_score))

        def train_classification(batch, step_i):
            loss_val, summary, acc,  _ = self.sess.run([task.loss, self.merged, task.acc, train_cls,
                                             ],
                                            feed_dict=batch2feed_dict(batch)
                                            )
            self.log.debug("Step {0} train loss={1:.04f} acc={2:.03f}".format(step_i, loss_val, acc))
            self.train_writer.add_summary(summary, self.g_step)
            return loss_val, acc

        def top_1_as_mask(np_arr):
            r = np.zeros_like(np_arr)
            r[np.argmax(np_arr)] = 1
            return r

        def over_zero(np_arr):
            return np.less(0, np_arr).astype(np.float32)

        multi_deletion = True
        if multi_deletion == True:
            logit2tag = over_zero
        else:
            logit2tag = top_1_as_mask

        def sample_size():
            if multi_deletion:
                prob = [(1,0.5), (2,0.2), (3,0.1), (4,0.1), (5,0.1)]
                v = random.random()
                for n, p in prob:
                    v -= p
                    if v < 0:
                        return n
                return 1
            else:
                return 1

        def numpy_print(arr):
            return "".join(["{0:.3f} ".format(v) for v in arr])


        def train_explain(batch, step_i):
            logits, conf_logit = self.sess.run([task.sout, task.conf_logits
                                             ],
                                            feed_dict=batch2feed_dict(batch)
                                            )
            x0, x1, x2, y  = batch

            instance_infos = []
            pred = np.argmax(logits, axis=1)
            compare_deletion_num = 20
            new_batches = []
            deleted_mask_list = []
            for i in range(len(logits)):
                if pred[i] == CONTRADICTION:
                    info = {}
                    info['init_logit'] = logits[i]
                    info['orig_input'] = (x0[i],x1[i],x2[i],y[i])
                    conf_tags = logit2tag(conf_logit[i])
                    self.log2.debug("CONF: {}".format(numpy_print(conf_logit[i])))
                    tag_size = np.count_nonzero(conf_tags)
                    if tag_size > 10:
                        self.log.debug("#conflict token={}".format(tag_size))

                    info['idx_delete_tagged'] = len(new_batches)
                    new_batches.append(token_delete(conf_tags, x0[i], x1[i], x2[i]))
                    deleted_mask_list.append(conf_tags)


                    indice_delete_random = []

                    #indice_delete_random.append(len(new_batches))
                    #x_list, delete_indice = sample_delete(conf_logit[i], x0[i], x1[i], x2[i])
                    #new_batches.append(x_list)
                    #deleted_indice_list.append(delete_indice)

                    for _ in range(compare_deletion_num):
                        tag_size = sample_size()
                        indice_delete_random.append(len(new_batches))
                        x_list, delete_mask = random_delete(tag_size, x0[i], x1[i], x2[i])
                        new_batches.append(x_list)
                        deleted_mask_list.append(delete_mask)

                    info['indice_delete_random'] = indice_delete_random
                    instance_infos.append(info)
            # Try deletions runs
            if len(new_batches) == 0:
                return
            alt_batches = get_batches_ex(new_batches, self.hparam.batch_size, 3)
            num_test_batches = len(alt_batches)
            alt_logits = []
            for batch in alt_batches:
                x0, x1, x2 = batch
                logits, = self.sess.run([task.sout, ],
                                               feed_dict={
                                                task.x_list[0]: x0,
                                                task.x_list[1]: x1,
                                                task.x_list[2]: x2,
                                               })

                alt_logits.append(logits)
            alt_logits = np.concatenate(alt_logits)

            def logit2ce(logit):
                return logit[2] - logit[0]


            reinforce_payload = []
            def reinforce(good_action, bad_action, input_x):
                pos_reward_indice = np.int_(np.logical_and(good_action, np.logical_not(bad_action)))
                neg_reward_indice = np.int_(np.logical_and(bad_action, np.logical_not(good_action)))
                loss_mask = -pos_reward_indice + neg_reward_indice
                x0,x1,x2,y = input_x

                self.log2.debug("Good: {}".format(good_action))
                self.log2.debug("Bad : {}".format(bad_action))
                self.log2.debug("Mask: {}".format(loss_mask))
                reinforce_payload.append((x0, x1, x2, y, loss_mask))

            def reinforce_one(good_action, input_x):
                pos_reward_indice = np.int_(good_action)
                loss_mask = -pos_reward_indice + np.ones_like(pos_reward_indice) * 0.1
                x0,x1,x2,y = input_x

                self.log2.debug("Good: {}".format(good_action))
                self.log2.debug("Mask: {}".format(loss_mask))
                reinforce_payload.append((x0, x1, x2, y, loss_mask))


            # calc reward
            target_ce_drop_list = []
            pos_win = 0
            pos_trial = 0
            for info in instance_infos:
                init_ce = logit2ce(info['init_logit'])
                target_ce = logit2ce(alt_logits[info['idx_delete_tagged']])
                input_x = info['orig_input']

                predicted_action = deleted_mask_list[info['idx_delete_tagged']]
                num_tag = np.count_nonzero(predicted_action)
                penalty = (num_tag - 1) * 0.1
                target_ce_drop = init_ce - target_ce - penalty
                target_ce_drop_list.append(target_ce_drop)
                self.log2.debug(
                    "target_ce_drop : {0:.4f}  n_token : {1}".format(target_ce_drop, num_tag))

                good_action = predicted_action
                best_ce_drop = target_ce_drop
                for idx_delete_random in info['indice_delete_random']:
                    comp_ce = logit2ce(alt_logits[idx_delete_random])
                    comp_ce_drop = init_ce - comp_ce
                    if comp_ce_drop > best_ce_drop :
                        best_ce_drop = comp_ce_drop
                        good_action = deleted_mask_list[idx_delete_random]

                reinforce_one(good_action, input_x)
                if target_ce_drop >= best_ce_drop:
                    pos_win += 1
                pos_trial += 1

            match_rate = pos_win / pos_trial
            self.log.debug("ce drop : {0:.4f}  suc_rate : {1:0.2f}".format(average(target_ce_drop_list), match_rate))
            #  update gradient
            def reinforce_commit():
                batches = get_batches_ex(reinforce_payload, self.hparam.batch_size, 5)
                for batch in batches:
                    x0, x1, x2, y, rf_mask = batch
                    _, rl_loss, conf_logits, verbose_loss = self.sess.run([train_rl, task.rl_loss,
                                                                                task.conf_logits,
                                                                                task.verbose_loss],
                                            feed_dict={
                                                task.x_list[0]: x0,
                                                task.x_list[1]: x1,
                                                task.x_list[2]: x2,
                                                task.y : y,
                                                task.rf_mask : rf_mask,
                                            })
            reinforce_commit()


        def train_ws_fn(batch, step_i):
            loss_val, acc = train_classification(batch, step_i)




            return loss_val, acc

        def train_ex_fn(batch, step_i):
            # normal train
            loss_val, acc = train_classification(batch, step_i)
            #  Fetch token label
            if f_train_ex:
                train_explain(batch, step_i)
            self.g_step += 1
            return loss_val, acc

        def valid_fn():
            loss_list = []
            acc_list = []
            eval()
            for batch in dev_batches[:100]:
                loss_val, summary, acc = self.sess.run([task.loss, self.merged, task.acc],
                                                  feed_dict=batch2feed_dict(batch)
                                                  )
                loss_list.append(loss_val)
                acc_list.append(acc)

                self.test_writer.add_summary(summary, self.g_step)
            self.log.info("Validation : loss={0:.04f} acc={1:.04f}".format(average(loss_list), average(acc_list)))

        def save_fn():
            self.save_model(exp_config.name, 100)

        train_batches, dev_batches = self.load_nli_data(data_loader)
        valid_freq = 25
        num_epochs = exp_config.num_epoch
        for i_epoch in range(num_epochs):
            loss, _ = epoch_runner(train_batches, train_ex_fn,
                                   valid_fn, valid_freq,
                                   save_fn, self.save_interval)



    def rank_adhoc(self, exp_config, data_loader, preload_id):
        tprint("train_adhoc")
        task = transformer_adhoc(self.hparam, data_loader.voca_size)

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        rerank_size = 1000
        tprint("Loading Model...")
        if preload_id is not None:
            name = preload_id[0]
            id = preload_id[1]
            if exp_config.load_names:
                self.load_model_white(name, id, exp_config.load_names)
            else:
                self.load_model_bert(name, id)
        batch_size = self.hparam.batch_size
        def batch2feed_dict(batch):
            x0, x1, x2, y = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y: y,
            }
            return feed_dict

        encoder_unit = data_loader.encoder_unit

        def get_score(query_docs_list):
            def eval(runs):
                data = []
                for entry in runs:
                    data.append((entry['input_ids'], entry['input_mask'], entry['segment_ids'], 0))
                tprint("Packing batches (batch_size={})".format(batch_size))
                batches = get_batches_ex(data, batch_size, 4)
                tprint("Runing neural network prediction (#batch={})".format(len(batches)))
                y_list = []
                ticker = TimeEstimator(len(batches), sample_size=20)
                for batch in batches:
                    y, = self.sess.run([task.logits, ],
                                       feed_dict=batch2feed_dict(batch)
                                       )
                    y_list.append(y)
                    ticker.tick()
                ys = np.concatenate(y_list)
                return ys


            with ProcessPoolExecutor(max_workers=8) as executor:
                use_pickle = False
                if not use_pickle:
                    enc_payload = []
                    for query, doc_list in query_docs_list:
                        payload = []
                        print("Doc encoding for query '{}'".format(query))
                        for doc_id, text in doc_list:
                            # future[List[dict]]
                            runs_future = executor.submit(encoder_unit.encode_long_text, query, text)
                            payload.append((doc_id, runs_future))

                        for doc_id, runs_future, in payload:
                            runs = runs_future.result()
                            enc_payload.append((query, doc_id, runs))

                    save_to_pickle(enc_payload, "enc_payload")
                else:
                    tprint("Loading from pickles")
                    enc_payload = load_from_pickle("enc_payload")

            tprint("Scheduling NN runs")
            pk = PromiseKeeper(eval)
            score_list_future = []
            for query, doc_id, runs in enc_payload:
                y_futures = list([MyPromise(x, pk).future() for x in runs])
                score_list_future.append((query, doc_id, y_futures))

            pk.do_duty()
            tprint("Completed GPU computations")
            per_query = defaultdict(list)
            for query, doc_id, y_futures in score_list_future:
                per_query[query].append((doc_id, sum_future(y_futures)))

            result = []
            for query, _ in query_docs_list:
                result.append(per_query[query])
            return result

        def bm25_run(target_queries):
            # query with BM25
            # rescore
            # compare score
            result = []
            for q in target_queries:
                score = Counter()
                for q_term in q.split():
                    if q_term in tf_index:
                        for doc_id, tf in tf_index[q_term].items():
                            score[doc_id] += tf * idf[q_term]

                top_docs = list(score.most_common(rerank_size))
                result.append((q, top_docs))
            return result

        collection, idf, inv_index, tf_index, queries = load_trec_data_proc()

        #queries = list(load_mobile_queries())
        unique_queries = list(set(right(queries)))
        tprint("{} unique queries".format(len(unique_queries)))
        for query in unique_queries:
            print(query)
        bm25_result = bm25_run(unique_queries)

        def rerank():
            fout = path.open_pred_output("rerank_{}".format(exp_config.name))
            tprint("rerank")
            pay_load = []
            for q, top_docs in bm25_result:
                docs = list([(doc_id, collection[doc_id]) for doc_id in left(top_docs)])
                pay_load.append((q, docs))

            nn_result = get_score(pay_load)
            q_rank = dict()
            for query_idx, q_result in enumerate(nn_result):
                q_result = list(q_result)
                q_result.sort(key=lambda x:x[1], reverse=True)
                print(q_result[:5])
                print(q_result[-5:])
                query = unique_queries[query_idx]
                q_rank[query] = q_result

            for query_id, query in queries:
                q_result = q_rank[query]
                rank_idx = 1
                for doc_id, score in q_result:
                    fout.write("{} Q0 {} {} {} galago\n".format(query_id, doc_id, rank_idx, score[0]))
                    rank_idx += 1
        rerank()

    def train_adhoc2(self, exp_config, data_loader, preload_id):
        tprint("train_adhoc2")
        task = transformer_adhoc(self.hparam, data_loader.voca_size)
        with tf.variable_scope("optimizer"):
            train_op = self.get_train_op(task.loss)
        self.log.name = exp_config.name
        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)
        batch_size = self.hparam.batch_size

        tprint("Loading Model...")
        if preload_id is not None:
            name = preload_id[0]
            id = preload_id[1]
            if exp_config.load_names:
                self.load_model_white(name, id, exp_config.load_names)
            else:
                self.load_model_bert(name, id)

        tprint("get_dev_data...")
        dev_batches = data_loader.get_dev_data()


        def valid_fn():
            #compare_bm25()
            loss_list = []
            for batch in dev_batches:
                loss_val, summary, = self.sess.run([task.loss, self.merged,],
                                                  feed_dict=batch2feed_dict(batch)
                                                  )
                loss_list.append(loss_val)

                self.test_writer.add_summary(summary, self.g_step)
            self.log.info("Validation : loss={0:.04f}".format(average(loss_list)))


        def train_fn(batch, step_i):
            # normal train
            loss_val, summary, _= self.sess.run([task.loss, self.merged, train_op],
                                               feed_dict=batch2feed_dict(batch)
                                               )
            self.log.debug("Step {0} train loss={1:.04f}".format(step_i, loss_val))
            self.train_writer.add_summary(summary, self.g_step)
            self.g_step += 1
            return loss_val, 0

        def batch2feed_dict(batch):
            x0, x1, x2, y = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
            }
            return feed_dict

        def save_fn():
            self.save_model(exp_config.name, 100)


        valid_freq = 25
        last_save = time.time()
        max_step = 1000 * 1000 * 1000
        for step_i in range(max_step):
            if step_i % valid_freq == 0:
                valid_fn()

            if save_fn is not None:
                if time.time() - last_save > self.save_interval:
                    save_fn()
                    last_save = time.time()

            batch = data_loader.get_train_batch()
            train_fn(batch, step_i)
            self.g_step += 1





    def test_ql(self, exp_config, data_loader, preload_id):
        task = transformer_ql(self.hparam, data_loader.voca_size)

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)

        print("Loading Model...")
        if preload_id is not None:
            name = preload_id[0]
            id = preload_id[1]
            if exp_config.load_names:
                self.load_model_white(name, id, exp_config.load_names)
            else:
                self.load_model_bert(name, id)

        encoder_unit = data_loader.encoder_unit

        collection = load_trec(trecText_path)

        keys = list(collection.keys())
        doc = collection[keys[0]]
        doc2 = collection[keys[1]]

        # generate query

        tokens = doc.split()

        queries = []
        enc_queries = []
        n_query = 30
        query_seq_len = 10
        for i in range(n_query):
            st = i * 10
            query = tokens[st:st+2]
            queries.append(query)
            query_str = " ".join(query)
            enc_query = encoder_unit.encoder.encode(query_str)
            enc_query = enc_query + (query_seq_len - len(enc_query)) * [0]
            q_mask = [1] * len(enc_query) + (query_seq_len - len(enc_query)) * [0]
            enc_queries.append((enc_query, q_mask))

        # encode doc
        data = []
        for enc_query, q_mask in enc_queries:
            for text in [doc, doc2]:
                tokens_a = encoder_unit.encoder.encode(text)
                encoded = encoder_unit.encode_inner(tokens_a, [])
                data.append((encoded['input_ids'],
                             encoded['input_mask'],
                             encoded['segment_ids'],
                             enc_query,
                             q_mask,
                             0))

        batch_size = n_query * 2
        b = get_batches_ex(data, batch_size, 6)

        batch = b[0]
        x0, x1, x2, q0, q1, y = batch
        score, = self.sess.run([task.ql_score, ],
                               feed_dict={
                                   task.x_list[0]: x0,
                                   task.x_list[1]: x1,
                                   task.x_list[2]: x2,
                                   task.query: q0,
                                   task.q_mask: q1,
                                   task.y: y,
                               }
                           )
        for i in range(n_query):
            with_q = score[i*2]
            wo_q = score[i*2+1]
            print(with_q, wo_q, with_q<wo_q)




    def rank_ql(self, exp_config, data_loader, preload_id):
        task = transformer_ql(self.hparam, data_loader.voca_size)
        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)
        print("Loading Model...")
        if preload_id is not None:
            name = preload_id[0]
            id = preload_id[1]
            if exp_config.load_names:
                self.load_model_white(name, id, exp_config.load_names)
            else:
                self.load_model_bert(name, id)

        encoder_unit = data_loader.encoder_unit
        batch_size = self.hparam.batch_size
        query_seq_len = self.hparam.query_seq_len
        rerank_size = 100 #1000
        def get_score(query_docs_list):
            def eval(runs):
                tprint("Packing batches (batch_size={})".format(batch_size))
                batches = get_batches_ex(runs, batch_size, 6)

                tprint("Runing neural network prediction (#batch={})".format(len(batches)))
                y_list = []
                ticker = TimeEstimator(len(batches), sample_size=20)
                for batch in batches:
                    x0, x1, x2, q0, q1, y = batch
                    score, = self.sess.run([task.ql_score, ],
                                           feed_dict={
                                               task.x_list[0]: x0,
                                               task.x_list[1]: x1,
                                               task.x_list[2]: x2,
                                               task.query: q0,
                                               task.q_mask: q1,
                                               task.y: y,
                                           })
                    y_list.append(score)
                    ticker.tick()
                ys = np.concatenate(y_list)
                return ys


            with ProcessPoolExecutor(max_workers=8) as executor:
                encoded_docs_f = {}
                for query, doc_list in query_docs_list:
                    tprint("Doc encoding for query '{}'".format(query))
                    for doc_id, text in doc_list:
                        # future[List[dict]]
                        runs_future = executor.submit(encoder_unit.encode_long_text_single, text)
                        if doc_id not in encoded_docs_f:
                            encoded_docs_f[doc_id] = runs_future


                encoded_docs = {}
                for doc_id, runs_future in encoded_docs_f.items():
                    runs = runs_future.result()
                    assert type(runs[0]) == dict
                    encoded_docs[doc_id] = runs
                tprint("... Done")

                save_to_pickle(encoded_docs, "encoded_docs")

            tprint("Scheduling NN runs")
            pk = PromiseKeeper(eval)
            score_list_future = []
            for query, doc_list in query_docs_list:
                enc_query = encoder_unit.encoder.encode(query)
                if len(enc_query) > query_seq_len:
                    print("WARNING!!! query len exceed : {}".format(len(enc_query)))

                enc_query = enc_query[:query_seq_len]
                q_mask = [1] * len(enc_query) + (query_seq_len - len(enc_query)) * [0]
                enc_query = enc_query + (query_seq_len - len(enc_query)) * [0]

                for doc_id, _ in doc_list:
                    y_futures = []
                    for encoded in encoded_docs[doc_id]:
                        run = (encoded['input_ids'],
                             encoded['input_mask'],
                             encoded['segment_ids'],
                             enc_query,
                             q_mask,
                             0)
                        y_futures.append(MyPromise(run, pk).future())

                    score_list_future.append((query, doc_id, y_futures))

            pk.do_duty()
            tprint("Completed GPU computations")
            per_query = defaultdict(list)
            for query, doc_id, y_futures in score_list_future:
                per_query[query].append((doc_id, max_future(y_futures)))

            result = []
            for query, _ in query_docs_list:
                result.append(per_query[query])
            return result

        def bm25_run(target_queries):
            # query with BM25
            # rescore
            # compare score
            result = []
            for q in target_queries:
                score = Counter()
                for q_term in q.split():
                    if q_term in tf_index:
                        for doc_id, tf in tf_index[q_term].items():
                            score[doc_id] += tf * idf[q_term]

                top_docs = list(score.most_common(rerank_size))
                result.append((q, top_docs))
            return result

        collection, idf, inv_index, tf_index, queries = load_trec_data_proc()

        queries = list(load_mobile_queries())
        unique_queries = list(set(right(queries)))
        tprint("{} unique queries".format(len(unique_queries)))
        bm25_result = bm25_run(unique_queries)

        def rerank():
            fout = path.open_pred_output("rerank_{}".format(exp_config.name))
            tprint("rerank")
            pay_load = []
            for q, top_docs in bm25_result:
                docs = list([(doc_id, collection[doc_id]) for doc_id in left(top_docs)])
                pay_load.append((q, docs))

            nn_result = get_score(pay_load)
            q_rank = dict()
            for query_idx, q_result in enumerate(nn_result):
                q_result = list(q_result)
                q_result.sort(key=lambda x:x[1], reverse=True)
                print(q_result[:5])
                print(q_result[-5:])
                query = unique_queries[query_idx]
                q_rank[query] = q_result

            for query_id, query in queries:
                q_result = q_rank[query]
                rank_idx = 1
                for doc_id, score in q_result:
                    fout.write("{} Q0    {} {} {} galago\n".format(query_id, doc_id, rank_idx, score))
                    rank_idx += 1
        rerank()


    def train_controversy_classification(self, exp_config, data_loader, preload_id):

        if exp_config.name.startswith("Contrv_B"):
            task = transformer_controversy(self.hparam, data_loader.voca_size)
            with tf.variable_scope("optimizer"):
                train_op = self.get_train_op(task.loss)
        elif exp_config.name.startswith("Contrv_C"):
            task = transformer_controversy(self.hparam, data_loader.voca_size)
            with tf.variable_scope("optimizer"):
                train_op = self.get_train_op_with_black_list(task.loss, 'bert/embeddings/word_embeddings')
        else:
            raise Exception("Undefined Model")

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)
        batch_size = self.hparam.batch_size
        self.log.name = exp_config.name

        print("Loading Model...")
        if preload_id is not None:
            name = preload_id[0]
            id = preload_id[1]
            if exp_config.load_names:
                self.load_model_white(name, id, exp_config.load_names)
            else:
                self.load_model_bert(name, id)



        def generate_train_batch():
            data = data_loader.get_train_data(batch_size)
            batches = get_batches_ex(data, batch_size, 3)
            return batches[0]

        print("Init queue feader ")
        max_reserve_batch = 100
        queue_feader = QueueFeader(max_reserve_batch, generate_train_batch)


        def save_fn():
            self.save_model(exp_config.name, 100)

        def batch2feed_dict(batch):
            x0, x1, x2  = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
            }
            return feed_dict

        def train_fn(batch, step_i):
            loss_val, summary, _ = self.sess.run([task.loss, self.merged, train_op,
                                             ],
                                            feed_dict=batch2feed_dict(batch)
                                            )
            self.log.debug("Step {0} train loss={1:.04f}".format(step_i, loss_val))
            self.train_writer.add_summary(summary, self.g_step)
            self.g_step += 1
            return loss_val, 0

        dev_data = data_loader.get_dev_data(batch_size * 10)
        dev_batch = get_batches_ex(dev_data, batch_size, 3)
        def valid_fn():
            loss_list = []
            for batch in dev_batch:
                loss_val, summary = self.sess.run([task.loss, self.merged],
                                          feed_dict=batch2feed_dict(batch)
                                          )

                self.test_writer.add_summary(summary, self.g_step)

                loss_list.append(loss_val)
            self.log.info("Validation : loss={0:.04f}".format(average(loss_list)))

        valid_freq = 25
        last_save = time.time()
        max_step = 1000 * 1000 * 1000

        for step_i in range(max_step):
            if step_i % valid_freq == 0:
                valid_fn()

            if save_fn is not None:
                if time.time() - last_save > exp_config.save_interval:
                    save_fn()
                    last_save = time.time()
                    if exp_config.save_interval < 120 * 60:
                        exp_config.save_interval += 60 * 10

            batch = queue_feader.get()

            train_fn(batch, step_i)
            self.g_step += 1



    def test_controversy_mscore(self,exp_config, data_loader, preload_id):
        task = transformer_controversy(self.hparam, data_loader.voca_size)

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)
        batch_size = self.hparam.batch_size

        print("Loading Model...")
        if preload_id is not None:
            name = preload_id[0]
            id = preload_id[1]
            if exp_config.load_names:
                self.load_model_white(name, id, exp_config.load_names)
            else:
                self.load_model_bert(name, id)

        labels = controversy.load_label()
        docs = controversy.load_docs()

        docs_dict = dict(docs)

        prior_cont = 0.35
        encoder_unit = data_loader.encoder_unit

        def get_score(docs):
            def eval(runs):
                tprint("Packing batches (batch_size={})".format(batch_size))
                batches = get_batches_ex(runs, batch_size, 3)

                tprint("Runing neural network prediction (#batch={})".format(len(batches)))
                y_list = []
                ticker = TimeEstimator(len(batches), sample_size=20)
                for batch in batches:
                    x0, x1, x2, = batch
                    score, = self.sess.run([task.logits, ],
                                           feed_dict={
                                               task.x_list[0]: x0,
                                               task.x_list[1]: x1,
                                               task.x_list[2]: x2,
                                           })
                    score = score.reshape([-1])
                    y_list.append(score)
                    ticker.tick()
                ys = np.concatenate(y_list)
                return ys


            with ProcessPoolExecutor(max_workers=8) as executor:
                encoded_docs_f = []
                for doc_id, text in docs:
                    # future[List[dict]]
                    runs_future = executor.submit(encoder_unit.encode_long_text_single, text)
                    encoded_docs_f.append(runs_future)

                encoded_docs = list([f.result() for f in encoded_docs_f])
                tprint("... Done")


            tprint("Scheduling NN runs")
            pk = PromiseKeeper(eval)
            score_list_future = []

            for doc_idx in range(len(docs)):
                y_futures = []
                doc_id = docs[doc_idx][0]
                for encoded in encoded_docs[doc_idx]:
                    run = (encoded['input_ids'],
                         encoded['input_mask'],
                         encoded['segment_ids'],
                         )
                    y_futures.append(MyPromise(run, pk).future())

                score_list_future.append((doc_id, y_futures))

            pk.do_duty()
            tprint("Completed GPU computations")
            result = []
            for doc_id, y_futures in score_list_future:
                result.append((doc_id, max_future(y_futures)))

            return result


        scores = get_score(docs)


        def compute_auc(y_scores, y_true):
            assert len(y_scores) == len(y_true)
            print(y_scores)
            p, r, thresholds = roc_curve(y_true, y_scores)
            return metrics.auc(p, r)

        golds = list([labels[d] for d, _ in docs])
        auc = compute_auc(right(scores), golds)

        scores.sort(key=lambda x:x[1], reverse=True)

        cut_idx = int(len(scores) * prior_cont)
        cont_docs = scores[:cut_idx]


        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for doc_id, score in cont_docs:
            text = docs_dict[doc_id]
            print("Cont : {}".format(text[:100]))
            print("Gold = {0} score = {1:.2f}".format(labels[doc_id], score))
            if labels[doc_id] == 1:
                tp += 1
            else:
                fp += 1

        for doc_id, score in scores[cut_idx:]:
            text = docs_dict[doc_id]
            print("NotCont : {}".format(text[:100]))
            print("Gold = {0} score = {1:.2f}".format(labels[doc_id], score))


            if labels[doc_id] == 0:
                tn += 1
            else:
                fn += 1

        prec = tp / (tp+fp)
        recall = tp / (tp + fn)
        acc = (tp+tn) / (tp+fp+tn+fn)
        f1 = 2*(prec*recall) / (prec+recall)
        print("-------")
        print("AUC\t", auc)
        print("prec\t", prec)
        print("recall\t", recall)
        print("F1\t", f1)
        print("acc\t", acc)



    def controv_lm(self, exp_config, data_loader, preload_id):
        task = transformer_ql(self.hparam, data_loader.voca_size)
        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)
        print("Loading Model...")
        if preload_id is not None:
            name = preload_id[0]
            id = preload_id[1]
            if exp_config.load_names:
                self.load_model_white(name, id, exp_config.load_names)
            else:
                self.load_model_bert(name, id)

        encoder_unit = data_loader.encoder_unit
        batch_size = self.hparam.batch_size
        query_seq_len = self.hparam.query_seq_len

        def get_score(docs):
            def eval(runs):
                tprint("Packing batches (batch_size={})".format(batch_size))
                batches = get_batches_ex(runs, batch_size, 6)

                tprint("Runing neural network prediction (#batch={})".format(len(batches)))
                y_list = []
                ticker = TimeEstimator(len(batches), sample_size=20)
                for batch in batches:
                    x0, x1, x2, q0, q1, y = batch
                    score, = self.sess.run([task.ql_score, ],
                                           feed_dict={
                                               task.x_list[0]: x0,
                                               task.x_list[1]: x1,
                                               task.x_list[2]: x2,
                                               task.query: q0,
                                               task.q_mask: q1,
                                               task.y: y,
                                           })
                    y_list.append(score)
                    ticker.tick()
                ys = np.concatenate(y_list)
                return ys


            with ProcessPoolExecutor(max_workers=8) as executor:
                encoded_docs_f = []
                for doc_id, text in docs:
                    # future[List[dict]]
                    runs_future = executor.submit(encoder_unit.encode_long_text_single, text)
                    encoded_docs_f.append(runs_future)

                encoded_docs = list([f.result() for f in encoded_docs_f])
                tprint("... Done")


            tprint("Scheduling NN runs")
            pk = PromiseKeeper(eval)
            score_list_future = []

            query = "controversy controversial dispute"
            def encode_query(query):
                enc_query = encoder_unit.encoder.encode(query)
                print(query)
                print(enc_query)
                enc_query = enc_query[:query_seq_len]
                q_mask = [1] * len(enc_query) + (query_seq_len - len(enc_query)) * [0]
                enc_query = enc_query + (query_seq_len - len(enc_query)) * [0]
                return enc_query, q_mask

            enc_query, q_mask = encode_query(query)

            for doc_idx in range(len(docs)):
                y_futures = []
                doc_id = docs[doc_idx][0]
                for encoded in encoded_docs[doc_idx]:
                    run = (encoded['input_ids'],
                         encoded['input_mask'],
                         encoded['segment_ids'],
                         enc_query,
                         q_mask,
                         0)
                    y_futures.append(MyPromise(run, pk).future())

                score_list_future.append((doc_id, y_futures))

            pk.do_duty()
            tprint("Completed GPU computations")

            result = []
            for doc_id, y_futures in score_list_future:
                result.append((doc_id, max_future(y_futures)))

            return result

        labels = controversy.load_label()
        docs = controversy.load_docs()

        docs_dict = dict(docs)

        prior_cont = 0.35

        scores = get_score(docs)
        #save_to_pickle(scores, "scores")
        #scores = load_from_pickle("scores")



        def compute_auc(y_scores, y_true):
            assert len(y_scores) == len(y_true)
            print(y_scores)
            p, r, thresholds = roc_curve(y_true, y_scores)
            return metrics.auc(p, r)

        golds = list([labels[d] for d, _ in docs])
        auc = compute_auc(right(scores), golds)

        scores.sort(key=lambda x:random.random(), reverse=True)

        cut_idx = int(len(scores) * prior_cont)
        cont_docs = scores[:cut_idx]


        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for doc_id, score in cont_docs:
            text = docs_dict[doc_id]
            print("Cont : {}".format(text[:100]))
            print("Gold = {0} score = {1:.2f}".format(labels[doc_id], score))
            if labels[doc_id] == 1:
                tp += 1
            else:
                fp += 1

        for doc_id, score in scores[cut_idx:]:
            text = docs_dict[doc_id]
            print("NotCont : {}".format(text[:100]))
            print("Gold = {0} score = {1:.2f}".format(labels[doc_id], score))


            if labels[doc_id] == 0:
                tn += 1
            else:
                fn += 1

        prec = tp / (tp+fp)
        recall = tp / (tp + fn)
        acc = (tp+tn) / (tp+fp+tn+fn)
        f1 = 2*(prec*recall) / (prec+recall)
        print("-------")
        print("AUC\t", auc)
        print("prec\t", prec)
        print("recall\t", recall)
        print("F1\t", f1)
        print("acc\t", acc)

