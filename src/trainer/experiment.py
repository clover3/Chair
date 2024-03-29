from concurrent.futures import ProcessPoolExecutor

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import cache
import data_generator.NLI.enlidef as ENLIDef
import data_generator.adhoc.score_loader as score_loader
from adhoc.bm25_ex import get_bm25
from attribution.baselines import *
from attribution.eval import eval_explain, eval_pairing, predict_translate
from attribution.eval import eval_fidelity
from attribution.lime import *
from cpath import get_model_full_path
from data_generator import shared_setting
from data_generator.adhoc.ws import *
from data_generator.argmining import ukp
from data_generator.data_parser import controversy, load_protest
from data_generator.data_parser.robust import *
from data_generator.data_parser.robust2 import load_2k_rank
from data_generator.data_parser.trec import *
from data_generator.mask_lm import enwiki
from data_generator.old_codes.cnn import SimpleLoader
from data_generator.stance import stance_detection
from data_generator.ubuntu import ubuntu
from evaluation import *
from explain import visualize
from explain.explain_trainer import ExplainTrainer
from list_lib import left, right
from log import log
from models import word2vec
from models.baselines import svm
from models.cnn import CNN
from models.controversy import get_wiki_doc
from models.transformer.ScoreCombiner import *
from models.transformer.cie import token_regression, span_selection
from models.transformer.hyperparams import HPMerger
from models.transformer.tranformer_nli import transformer_nli, transformer_nli_embedding_in, transformer_nli_vector, \
    transformer_nli_e
from models.transformer.transformer_adhoc import transformer_adhoc, transformer_adhoc2, transformer_adhoc_ex
from models.transformer.transformer_arg_dist import transformer_distribution
from models.transformer.transformer_binary import transformer_binary
from models.transformer.transformer_concat import transformer_concat
from models.transformer.transformer_controversy import transformer_controversy
from models.transformer.transformer_lm import transformer_ql
from models.transformer.transformer_multitask import transformer_mt
from models.transformer.transformer_paired import transformer_paired
from models.transformer.transformer_weight import transformer_weight
from task.AuxLM import AuxLM
from task.MLM import TransformerLM
from task.PairFeature import PairFeature, PairFeatureClassification
from task.PairLM import TransformerPairLM
from task.aux_classification import AuxClassification
from task.classification import TransformerClassifier
from task.consistent_classification import ConsistentClassifier
from task.crs_classifier import crs_transformer as CrsModel
from task.metrics import eval_3label, eval_2label
from task.metrics import stance_f1
from trainer.np_modules import *
from trainer.promise import *
from trainer.queue_feader import QueueFeader


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
        self.model_dir = cpath.common_model_dir_root
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
        #self.global_step = tf.Variable(0, name='global_step', trainable=True)
        self.global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(learning_rate=self.hparam.lr, beta1=0.9, beta2=0.98, epsilon=1e-8,
                                           name=name)
        train_op = optimizer.minimize(loss, global_step=self.global_step)
        return train_op

    def get_train_op2(self, loss, lr, name='Adam'):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.98, epsilon=1e-8,
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
        max_sequence = 140
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
        summary_path = os.path.join(cpath.output_path, "summary")
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
        return ret


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

    def load_model_white(self, name, id, include_namespace, verbose=True):
        run_dir = os.path.join(self.model_dir, 'runs')
        save_dir = os.path.join(run_dir, name)
        path = os.path.join(save_dir, "{}".format(id))
        if verbose:
            print("Restoring: {} {}".format(name, id))

        self.load_model_white_fullpath(path, include_namespace, verbose)


    def load_model_white_fullpath(self, path, include_namespace, verbose=True):
        def condition(v):
            if v.name.split('/')[0] in include_namespace:
                return True
            return False

        variables = tf.contrib.slim.get_variables_to_restore()
        variables_to_restore = [v for v in variables if condition(v)]
        if verbose:
            for v in variables_to_restore:
                print(v)

        self.loader = tf.train.Saver(variables_to_restore, max_to_keep=1)
        self.loader.restore(self.sess, path)

    def load_model_white2(self, preload_id, include_namespace, verbose=True):
        if preload_id is None:
            return
        name = preload_id[0]
        id = preload_id[1]
        self.load_model_white(name, id, include_namespace, verbose)

    def preload_id_to_path(self, preload_id):
        name = preload_id[0]
        id = preload_id[1]
        run_dir = os.path.join(self.model_dir, 'runs')
        save_dir = os.path.join(run_dir, name)
        path = os.path.join(save_dir, "{}".format(id))
        return path


    def load_model_v2_map(self, preload_id, include_namespace, verbose=True):
        path = self.preload_id_to_path(preload_id)

        def condition(v):
            return condition_n(v.name)

        def condition_n(name):
            if name.split('/')[0] in include_namespace:
                return True
            return False

        tvars = tf.contrib.slim.get_variables_to_restore()
        name_to_variable = collections.OrderedDict()
        for var in tvars:
            name = var.name
            m = re.match("^(.*):\\d+$", name)
            if m is not None:
                name = m.group(1)
            name_to_variable[name] = var

        init_vars = tf.train.list_variables(path)

        load_mapping = dict()
        for v in init_vars:
            if condition_n(v[0]):
                name_tokens = v[0].split('/')
                var_name = '/'.join(name_tokens).split(":")[0]
                print(var_name)
                targ_name = re.sub("layer_normalization[_]?\d*", "LayerNorm", var_name)
                targ_name = re.sub("dense[_]?\d*", "dense", targ_name)
                if targ_name in name_to_variable:
                    if verbose:
                        print(targ_name)
                    load_mapping[var_name] = name_to_variable[targ_name]


        self.loader = tf.train.Saver(load_mapping, max_to_keep=1)
        self.loader.restore(self.sess, path)


    def load_concat_model(self, load_id_list, include_namespace, target_namespaces):
        def get_path(preload_id):
            name = preload_id[0]
            id = preload_id[1]
            run_dir = os.path.join(self.model_dir, 'runs')
            save_dir = os.path.join(run_dir, name)
            path = os.path.join(save_dir, "{}".format(id))
            return path

        def condition(v):
            if v.name.split('/')[1] in include_namespace:
                return True
            return False

        for load_id, cur_target_namespace in zip(load_id_list, target_namespaces):
            path = get_path(load_id)
            variables = tf.contrib.slim.get_variables_to_restore()
            variables_to_restore = [v for v in variables if condition(v)]
            load_mapping = dict()
            for v in variables_to_restore:
                name_tokens = v.name.split('/')
                if name_tokens[0] == cur_target_namespace :
                    save_name = '/'.join(name_tokens[1:]).split(":")[0]
                    load_mapping[save_name] = v
                    #print("Mapping {} -> {}".format(save_name, v.name))
            loader = tf.train.Saver(load_mapping, max_to_keep=1)
            loader.restore(self.sess, path)

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

    def pickle_cacher(self, pickle_name, get_fn, use_pickle):
        pickle_path = os.path.join(cpath.common_model_dir_root, pickle_name)
        print(pickle_path)
        if not use_pickle:
            obj = get_fn()
            pickle.dump(obj, open(pickle_path, "wb"))
        else:
            obj = pickle.load(open(pickle_path, "rb"))
        return obj

    def load_nli_data(self, data_loader):
        # load data
        print("Loading Data")

        pickle_name = "{}_{}".format(data_loader.name, self.hparam.batch_size)

        def get_data():
            train_batches = get_batches_ex(data_loader.get_train_data(), self.hparam.batch_size, 4)
            dev_batches = get_batches_ex(data_loader.get_dev_data(), self.hparam.batch_size, 4)
            return train_batches, dev_batches

        return self.pickle_cacher(pickle_name, get_data, use_pickle=True)

    def load_nli_data_with_info(self, data_loader):
        print("Loading Data")
        pickle_name = "nli_plus_{}".format(self.hparam.batch_size)

        def get_data():
            train_batches, dev_batches = self.load_nli_data(data_loader)
            train_batches_info = list_batch_grouping(data_loader.get_train_infos(), self.hparam.batch_size)

            assert len(train_batches) == len(train_batches_info)
            return train_batches, train_batches_info, dev_batches

        return self.pickle_cacher(pickle_name, get_data, use_pickle=True)

    def train_nli(self, nli_setting, exp_config, data_loader):
        print("train_nli")
        task = transformer_nli(self.hparam, nli_setting.vocab_size, 1)
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

        def save_fn():
            self.save_model(exp_config.name, 10)

        valid_freq = 100
        # train_op
        print("Train epoch")
        num_epochs = exp_config.num_epoch
        for i_epoch in range(num_epochs):
            loss, _ = epoch_runner(train_batches, train_fn,
                                   valid_fn, valid_freq,
                                   save_fn, self.save_interval)

        steps = int(len(train_batches) * 0.5)
        loss, _ = sub_step_runner(train_batches, train_fn,
                                  valid_fn, valid_freq,
                                  save_fn, self.save_interval,
                                  steps=steps)
        return save_fn()

    def nli_attribution_baselines(self, nli_setting, exp_config, data_loader, preload_id):
        print("attribution_explain")
        target = 'conflict'
        enc_explain_dev, explain_dev = data_loader.get_dev_explain(target)
        #train_batches, dev_batches = self.load_nli_data(data_loader
        from attribution.gradient import explain_by_gradient
        self.sess = self.init_sess()
        from attribution.deepexplain.tensorflow import DeepExplain
        with DeepExplain(session=self.sess, graph=self.sess.graph) as de:
            task = transformer_nli_embedding_in(self.hparam, nli_setting.vocab_size, False)
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


            emb_outputs = task.encoded_embedding_out, task.attention_mask_out
            emb_input = task.encoded_embedding_in, task.attention_mask_in
            softmax_out = task.sout
            def feed_end_input(batch):
                x0, x1, x2 = batch
                return {task.x_list[0]:x0,
                        task.x_list[1]:x1,
                        task.x_list[2]:x2,
                        }

            explain_prediction = {}
            elapsed_time = {}
            #method_list = [ "elrp", "deeplift", "saliency","grad*input", "intgrad", ]
            method_list = ["saliency", "grad*input", "intgrad", ]
            for method_name in method_list:
                print(method_name)
                begin = time.time()

                explains = explain_by_gradient(enc_explain_dev, method_name, target, self.sess, de,
                                               feed_end_input, emb_outputs, emb_input, softmax_out)
                end = time.time()
                elapsed_time[method_name] = end - begin
                explain_prediction[method_name] = explains

                print(method_name)
                print("Elapsed Time\t{}".format(elapsed_time[method_name]))
                explains = explain_prediction[method_name]
                scores = eval_explain(explains, data_loader, target)
                p_at_1, MAP_score, auc_score = scores["P@1"], scores["MAP"], scores["AUC"]
                print("P@1\t{}".format(p_at_1))
                print("MAP\t{}".format(MAP_score))
                print("AUC\t{}".format(auc_score))

    def nli_attribution_predict(self, nli_setting, exp_config, data_loader, preload_id, target_label, data_id):
#        enc_payload, plain_payload = data_loader.get_test_data(data_id)
        enc_payload, plain_payload = data_loader.load_plain_text(data_id)

        from attribution.gradient import explain_by_gradient
        self.sess = self.init_sess()
        from attribution.deepexplain.tensorflow import DeepExplain
        with DeepExplain(session=self.sess, graph=self.sess.graph) as de:
            task = transformer_nli_embedding_in(self.hparam, nli_setting.vocab_size, False)
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


            emb_outputs = task.encoded_embedding_out, task.attention_mask_out
            emb_input = task.encoded_embedding_in, task.attention_mask_in
            softmax_out = task.sout
            def feed_end_input(batch):
                x0, x1, x2 = batch
                return {task.x_list[0]:x0,
                        task.x_list[1]:x1,
                        task.x_list[2]:x2,
                        }

            #method_list = [ "elrp", "deeplift", "saliency","grad*input", "intgrad", ]
            method_list = ["intgrad","grad*input",  "saliency", ]
            for method_name in method_list:
                print(method_name)

                explains = explain_by_gradient(enc_payload, method_name, target_label, self.sess, de,
                                               feed_end_input, emb_outputs, emb_input, softmax_out)

                pred_list = predict_translate(explains, data_loader, enc_payload, plain_payload)
                save_to_pickle(pred_list, "pred_{}_{}".format(method_name, data_id))

    def nli_baseline_predict(self, nli_setting, exp_config, data_loader, preload_id,
                            explain_tag, data_id, sub_range):
        enc_payload, plain_payload = data_loader.load_plain_text(data_id)

        name_format = "pred_{}_" + data_id
        if sub_range is not None:
            st, ed = sub_range
            enc_payload = enc_payload[st:ed]
            plain_payload = plain_payload[st:ed]
            name_format = "pred_{}_" + data_id + "__{}_{}".format(st,ed)
            print(name_format)

        task = transformer_nli(self.hparam, nli_setting.vocab_size, 1)
        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.load_model_white2(preload_id, exp_config.load_names, False)
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

        #train_batches, dev_batches = self.load_nli_data(data_loader)
        #idf_scorer = IdfScorer(train_batches)

        todo_list = [
               #     ('deletion_seq', explain_by_seq_deletion),
               #     ('random', explain_by_random),
               #     ('idf', idf_scorer.explain),
               #     ('deletion', explain_by_deletion),
                    ('LIME', explain_by_lime),
                 ]
        for method_name, method in todo_list:
            explains = method(enc_payload, explain_tag, forward_run)
            pred_list = predict_translate(explains, data_loader, enc_payload, plain_payload)
            save_to_pickle(pred_list, name_format.format(method_name))

    def predict_lime_snli(self, nli_setting, exp_config, data_loader, preload_id,
                            data_id, sub_range):
        enc_payload, plain_payload = data_loader.load_plain_text(data_id)

        task = transformer_nli(self.hparam, nli_setting.vocab_size, 1)
        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.load_model_white2(preload_id, exp_config.load_names, False)

        name_format = "pred_{}_" + data_id


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

        step = 50
        #for st in range(0,3000, step):
        st,ed = sub_range
        #ed = st + step
        enc_payload = enc_payload[st:ed]
        plain_payload = plain_payload[st:ed]
        name_format = "pred_{}_" + data_id + "__{}_{}".format(st,ed)

        todo_list = [
                    ('LIME', explain_by_lime_notag),
                 ]
        for method_name, method in todo_list:
            explains = method(enc_payload, forward_run)
            pred_list = predict_translate(explains, data_loader, enc_payload, plain_payload)
            save_to_pickle(pred_list, name_format.format(method_name))

    def predict_lime_snli_continue(self, nli_setting, exp_config, data_loader, preload_id,
                          data_id):
        enc_payload, plain_payload = data_loader.load_plain_text(data_id)
        print("total : ", len(enc_payload))

        task = transformer_nli(self.hparam, nli_setting.vocab_size, 1)
        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.load_model_white2(preload_id, exp_config.load_names, False)


        def forward_run(inputs):
            batches = get_batches_ex(inputs, self.hparam.batch_size, 3)
            logit_list = []
            for batch in batches:
                x0, x1, x2 = batch
                soft_out, = self.sess.run([task.sout, ],
                                          feed_dict={
                                              task.x_list[0]: x0,
                                              task.x_list[1]: x1,
                                              task.x_list[2]: x2,
                                          })
                logit_list.append(soft_out)
            return np.concatenate(logit_list)

        step = 50
        for st in range(100, 3000, step):
            ed = st + step
            print(st, ed)
            enc_payload_slice = enc_payload[st:ed]
            plain_payload = plain_payload[st:ed]
            name_format = "pred_{}_" + data_id + "__{}_{}".format(st,ed)

            todo_list = [
                        ('LIME', explain_by_lime_notag),
                     ]
            assert enc_payload_slice
            for method_name, method in todo_list:
                explains = method(enc_payload_slice, forward_run)
                pred_list = predict_translate(explains, data_loader, enc_payload_slice, plain_payload)
                save_to_pickle(pred_list, name_format.format(method_name))


    def nli_explain_baselines(self, nli_setting, exp_config, data_loader, preload_id, explain_tag):
        enc_explain_dev, explain_dev = data_loader.get_dev_explain(explain_tag)

        enc_explain_dev = enc_explain_dev[:3]
        explain_dev = explain_dev[:3]
        #train_batches, dev_batches = self.load_nli_data(data_loader)
        paired_info = data_loader.match_explain_info(enc_explain_dev, explain_dev)


        task = transformer_nli(self.hparam, nli_setting.vocab_size, 1)

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
            print("Forward run with batchsize : ", self.hparam.batch_size)
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


        #idf_scorer = IdfScorer(train_batches)

        def eval(method):
            print("Eval")
            begin = time.time()
            explains = method(enc_explain_dev, explain_tag, forward_run)
            assert len(explains) == len(explain_dev)
            end = time.time()
            print("Elapsed Time : {}".format(end- begin))

            scores = eval_explain(explains, data_loader, explain_tag)
            for metric in scores.keys():
                print("{}\t{}".format(metric, scores[metric]))

        todo_list = [
                    #('deletion_seq', explain_by_seq_deletion),
                    #('random', explain_by_random),
                    #('idf', idf_scorer.explain),
                    #('deletion', explain_by_deletion),
                    ('lime', explain_by_lime),
                 ]
        for method_name, method in todo_list:
            print(method_name)
            eval(method)


    def train_nli_ex_0(self, nli_setting, exp_config, data_loader, preload_id, f_train_ex):
        if f_train_ex:
            enc_explain_dev, explain_dev = data_loader.get_dev_explain_0()
        target = 'conflict'
        task = transformer_nli(self.hparam, nli_setting.vocab_size, 2)
        with tf.variable_scope("optimizer"):
            train_cls = self.get_train_op(task.loss)
            train_rl = self.get_train_op(task.rl_loss, name="rl")
        print("Training ", data_loader.name)
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

            scores = eval_explain(conf_logit, data_loader, target)
            p_at_1, MAP_score = scores["P@1"], scores["MAP"]
            print("P@1 : {}".format(p_at_1))
            print("MAP : {}".format(MAP_score))
            summary = tf.Summary()
            summary.value.add(tag='P@1', simple_value=p_at_1)
            summary.value.add(tag='MAP', simple_value=MAP_score)
            self.train_writer.add_summary(summary, self.g_step)
            self.train_writer.flush()

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
                #prob = [(1,0.9), (2,0.1)]
                v = random.random()
                for n, p in prob:
                    v -= p
                    if v < 0:
                        return n
                return 1
            else:
                return 1


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
            tag_size_list = []
            for i in range(len(logits)):
                if pred[i] == CONTRADICTION:
                    info = {}
                    info['init_logit'] = logits[i]
                    info['orig_input'] = (x0[i],x1[i],x2[i],y[i])
                    conf_tags = logit2tag(conf_logit[i])
                    self.log2.debug("CONF: {}".format(numpy_print(conf_logit[i])))
                    tag_size = np.count_nonzero(conf_tags)
                    tag_size_list.append(tag_size)
                    if tag_size > 10:
                        self.log2.debug("#conflict token={}".format(tag_size))

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

            avg_tag_size = average(tag_size_list)
            self.log2.debug("avg tagged token#={}".format(avg_tag_size))

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
                reinforce_payload.append((x0, x1, x2, y, loss_mask))

            def supervise(good_action, input_x):
                label = np.int_(good_action)
                x0, x1, x2, y = input_x
                self.log2.debug("Label: {}".format(label))
                reinforce_payload.append((x0, x1, x2, y, label))

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
                #self.log2.debug(
                #    "target_ce_drop : {0:.4f}  n_token : {1}".format(target_ce_drop, num_tag))

                good_action = predicted_action
                best_ce_drop = target_ce_drop
                for idx_delete_random in info['indice_delete_random']:
                    comp_ce = logit2ce(alt_logits[idx_delete_random])
                    comp_ce_drop = init_ce - comp_ce
                    if comp_ce_drop > best_ce_drop :
                        best_ce_drop = comp_ce_drop
                        good_action = deleted_mask_list[idx_delete_random]

                #reinforce_one(good_action, input_x)
                supervise(good_action, input_x)
                if target_ce_drop >= best_ce_drop:
                    pos_win += 1
                pos_trial += 1

            match_rate = pos_win / pos_trial
            avg_ce_drop = average(target_ce_drop_list)
            self.log.debug("ce drop : {0:.4f}  suc_rate : {1:0.2f}".format(avg_ce_drop, match_rate))
            summary = tf.Summary()
            summary.value.add(tag='CE_Drop', simple_value=avg_ce_drop)
            summary.value.add(tag='Success', simple_value=match_rate)
            if tag_size_list:
                summary.value.add(tag='Tag Size', simple_value=avg_tag_size)
            self.train_writer.add_summary(summary, self.g_step)
            self.train_writer.flush()

            #  update gradient
            def reinforce_commit():
                batches = get_batches_ex(reinforce_payload, self.hparam.batch_size, 5)
                rl_loss_list = []
                for batch in batches:
                    x0, x1, x2, y, rf_mask = batch
                    _, rl_loss, conf_logits,  = self.sess.run([train_rl, task.rl_loss,
                                                                                task.conf_logits,
                                                                                ],
                                            feed_dict={
                                                task.x_list[0]: x0,
                                                task.x_list[1]: x1,
                                                task.x_list[2]: x2,
                                                task.y : y,
                                                task.rf_mask : rf_mask,
                                            })
                    rl_loss_list.append(rl_loss)
                summary.value.add(tag='RL_Loss', simple_value=average(rl_loss_list))
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
            if f_train_ex:
                eval()
            for batch in dev_batches[:100]:
                loss_val, summary, acc = self.sess.run([task.loss, self.merged, task.acc],
                                                  feed_dict=batch2feed_dict(batch)
                                                  )
                loss_list.append(loss_val)
                acc_list.append(acc)

                self.test_writer.add_summary(summary, self.g_step)
            self.log.info("Validation : loss={0:.04f} acc={1:.04f}".format(average(loss_list), average(acc_list)))
            return average(acc_list)

        def save_fn():
            return self.save_model(exp_config.name, 2)

        train_batches, dev_batches = self.load_nli_data(data_loader)

        print("{} train batches".format(len(train_batches)))
        valid_freq = 10000
        num_epochs = exp_config.num_epoch
        for i_epoch in range(num_epochs):
            loss, _ = epoch_runner(train_batches, train_ex_fn,
                                   valid_fn, valid_freq,
                                   save_fn, self.save_interval)

        steps = int(len(train_batches) * 0.5)
        loss, _ = sub_step_runner(train_batches, train_ex_fn,
                                  valid_fn, valid_freq,
                                  save_fn, self.save_interval,
                                  steps=steps)

        acc =valid_fn()
        print("Last val acc : ", acc)
        return save_fn()



    def train_nli_only_new(self, nli_setting, exp_config, data_loader, preload_id):
        task = transformer_nli_e(self.hparam, nli_setting.vocab_size, 2)
        with tf.variable_scope("optimizer"):
            train_cls = self.get_train_op(task.loss)
        print("Training ", data_loader.name)
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


        def train_classification(batch, step_i):
            loss_val, summary, acc,  _ = self.sess.run([task.loss, self.merged, task.acc, train_cls,
                                             ],
                                            feed_dict=batch2feed_dict(batch)
                                            )
            self.log.debug("Step {0} train loss={1:.04f} acc={2:.03f}".format(step_i, loss_val, acc))
            self.train_writer.add_summary(summary, self.g_step)
            return loss_val, acc



        def train_ex_fn(batch, step_i):
            # normal train
            loss_val, acc = train_classification(batch, step_i)
            self.g_step += 1
            return loss_val, acc

        def valid_fn():
            loss_list = []
            acc_list = []
            for batch in dev_batches[:100]:
                loss_val, summary, acc = self.sess.run([task.loss, self.merged, task.acc],
                                                  feed_dict=batch2feed_dict(batch)
                                                  )
                loss_list.append(loss_val)
                acc_list.append(acc)

                self.test_writer.add_summary(summary, self.g_step)
            self.log.info("Validation : loss={0:.04f} acc={1:.04f}".format(average(loss_list), average(acc_list)))
            return average(acc_list)

        def save_fn():
            return self.save_model(exp_config.name, 2)

        train_batches, dev_batches = self.load_nli_data(data_loader)

        print("{} train batches".format(len(train_batches)))
        valid_freq = 10000
        num_epochs = exp_config.num_epoch
        for i_epoch in range(num_epochs):
            loss, _ = epoch_runner(train_batches, train_ex_fn,
                                   valid_fn, valid_freq,
                                   save_fn, self.save_interval)

        steps = int(len(train_batches) * 0.5)
        loss, _ = sub_step_runner(train_batches, train_ex_fn,
                                  valid_fn, valid_freq,
                                  save_fn, self.save_interval,
                                  steps=steps)

        acc =valid_fn()
        print("Last val acc : ", acc)
        return save_fn()



    # Refactored version of explain train
    def test_acc2(self, nli_setting, exp_config, data_loader, save_path):
        method = 6
        task = transformer_nli(self.hparam, nli_setting.vocab_size, method, False)

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)
        self.load_model_white_fullpath(save_path, exp_config.load_names)

        def batch2feed_dict(batch):
            x0, x1, x2, y  = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y: y,
            }
            return feed_dict

        train_batches, dev_batches = self.load_nli_data(data_loader)
        def valid_fn():
            loss_list = []
            acc_list = []
            for batch in dev_batches:
                loss_val, summary, acc = self.sess.run([task.loss, self.merged, task.acc],
                                                  feed_dict=batch2feed_dict(batch)
                                                  )
                loss_list.append(loss_val)
                acc_list.append(acc)

                self.test_writer.add_summary(summary, self.g_step)
            print("Validation : loss={0:.04f} acc={1:.04f}".format(average(loss_list), average(acc_list)))
        print(save_path)
        valid_fn()


    # Refactored version of explain train
    def test_acc(self, nli_setting, exp_config, data_loader, preload_id):
        method = 6
        task = transformer_nli(self.hparam, nli_setting.vocab_size, method, False)

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)

        self.load_model_white2(preload_id, exp_config.load_names)

        def batch2feed_dict(batch):
            x0, x1, x2, y  = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y: y,
            }
            return feed_dict

        train_batches, dev_batches = self.load_nli_data(data_loader)
        def valid_fn():
            loss_list = []
            acc_list = []
            for batch in dev_batches:
                loss_val, summary, acc = self.sess.run([task.loss, self.merged, task.acc],
                                                  feed_dict=batch2feed_dict(batch)
                                                  )
                loss_list.append(loss_val)
                acc_list.append(acc)

                self.test_writer.add_summary(summary, self.g_step)
            print("Validation : loss={0:.04f} acc={1:.04f}".format(average(loss_list), average(acc_list)))
        print(preload_id)
        valid_fn()

    def nli_interactive(self, nli_setting, exp_config, data_loader, preload_id):
        task = transformer_nli(self.hparam, nli_setting.vocab_size, 1, False)

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)

        self.load_model_white2(preload_id, exp_config.load_names)

        def batch2feed_dict(batch):
            x0, x1, x2, y = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y: y,
            }
            return feed_dict

        train_batches, dev_batches = self.load_nli_data(data_loader)

        def predict(sents):
            data = []
            for sent1, sent2 in sents:
                entry = data_loader.encode(sent1, sent2)
                l = entry["input_ids"], entry["input_mask"], entry["segment_ids"], 0
                data.append(l)

            batches = get_batches_ex(data, self.hparam.batch_size, 4)
            logits, = self.sess.run([task.sout], feed_dict=batch2feed_dict(batches[0]))
            return logits

        terminate = False
        sents = []
        while not terminate:
            msg = input("Enter:")
            if msg == "!EOI":
                r = predict([(sents[0], sents[1]), (sents[1], sents[0])])
                print(r)
                sents = []
            elif msg == "!EXIT":
                terminate = True
            else:
                sents.append(msg)

        print(preload_id)


    def nli_interactive_list(self, nli_setting, exp_config, data_loader, preload_id):
        task = transformer_nli(self.hparam, nli_setting.vocab_size, 1, False)

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)

        self.load_model_white2(preload_id, exp_config.load_names)

        def batch2feed_dict(batch):
            x0, x1, x2, y = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y: y,
            }
            return feed_dict

        train_batches, dev_batches = self.load_nli_data(data_loader)

        def predict(sents):
            data = []
            for sent1, sent2 in sents:
                entry = data_loader.encode(sent1, sent2)
                l = entry["input_ids"], entry["input_mask"], entry["segment_ids"], 0
                data.append(l)

            batches = get_batches_ex(data, self.hparam.batch_size, 4)
            logit_list = []
            for batch in batches:
                logits, = self.sess.run([task.sout], feed_dict=batch2feed_dict(batch))
                logit_list.append(logits)

            logits = np.concatenate(logit_list)
            return logits

        terminate = False

        msg = ""
        while not terminate:
            sents = []
            msg = input("Enter:")
            while msg != "!EOI":
                sents.append(msg)
                msg = input("Enter:")


            #sents = msg.split("\n")
            n = len(sents)
            print("{} lines".format(n))



            payload = []
            info = []
            for i in range(n):
                for j in range(n):
                    if i== j:
                        continue
                    payload.append((sents[i], sents[j]))


            r = predict(payload)

            assert len(r) == len(payload)

            pred = r.argmax(axis=1)
            for i in range(len(payload)):
                if pred[i] != 1:
                    s1, s2 = payload[i]
                    print("\n")
                    print(s1)
                    print(s2)
                    label = {0:"Ent", 2:"Contradiction"}
                    print(label[pred[i]])



    def nli_interactive_visual(self, nli_setting, exp_config, data_loader, preload_id):
        task = transformer_nli(self.hparam, nli_setting.vocab_size, 1, False)

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)

        self.load_model_white2(preload_id, exp_config.load_names)

        def batch2feed_dict(batch):
            x0, x1, x2, y = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y: y,
            }
            return feed_dict

        def predict(sents):
            data = []
            for sent1, sent2 in sents:
                entry = data_loader.encode(sent1, sent2)
                l = entry["input_ids"], entry["input_mask"], entry["segment_ids"], 0
                data.append(l)

            batches = get_batches_ex(data, self.hparam.batch_size, 4)
            logits, conf = self.sess.run([task.sout, task.conf_logits], feed_dict=batch2feed_dict(batches[0]))
            predictions = logits.argmax(axis=1)
            x0, x1, x2, y = batches[0]
            result = []
            for idx in range(len(x0)):
                input_ids = x0[idx]
                conf_p, conf_h = data_loader.split_p_h_with_input_ids(conf[idx], input_ids)
                #prem, hypo, p_indice, h_indice = entry

                p_enc, h_enc = data_loader.split_p_h_with_input_ids(input_ids, input_ids)
                p_tokens = data_loader.encoder.decode_list(p_enc)
                h_tokens = data_loader.encoder.decode_list(h_enc)
                result.append((conf_p, conf_h, p_tokens, h_tokens, predictions[idx], y[idx]))
            visualize.visual_stream(result)
            return logits

        terminate = False
        sents = []
        while not terminate:
            msg = input("Enter:")
            if msg == "!EOI":
                r = predict([(sents[0], sents[1]), (sents[1], sents[0])])
                print(r)
                sents = []
            elif msg == "!EXIT":
                terminate = True
            else:
                sents.append(msg)

        print(preload_id)


    # Refactored version of explain train
    def predict_rf(self, nli_setting, exp_config, data_loader, preload_id, data_id, method=1):
        task = transformer_nli(self.hparam, nli_setting.vocab_size, method, False)

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)

        self.load_model_white2(preload_id, exp_config.load_names)

        def batch2feed_dict(batch):
            x0, x1, x2, y  = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y: y,
            }
            return feed_dict

        #train_batches, dev_batches = self.load_nli_data(data_loader)
        print(preload_id)

        enc_payload, plain_payload = data_loader.load_plain_text(data_id)

        batches = get_batches_ex(enc_payload, self.hparam.batch_size, 3)

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

        pred_list = predict_translate(conf_logit, data_loader, enc_payload, plain_payload)
        save_to_pickle(pred_list, "pred_{}_{}".format(exp_config.name, data_id))



    # Refactored version of explain train
    def train_nli_ex_1(self, nli_setting, exp_config, data_loader, preload_id, explain_tag):
        enc_explain_dev, explain_dev = data_loader.get_dev_explain(explain_tag)
        method = 5
        task = transformer_nli(self.hparam, nli_setting.vocab_size, method)
        with tf.variable_scope("optimizer"):
            train_cls = self.get_train_op(task.loss)
            train_rl = self.get_train_op(task.rl_loss, name="rl")

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)

        self.load_model_white2(preload_id, exp_config.load_names)

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

            scores = eval_explain(conf_logit, data_loader, explain_tag)
            p_at_1, MAP_score = scores["P@1"], scores["MAP"]
            print("P@1 : {}".format(p_at_1))
            print("MAP : {}".format(MAP_score))
            summary = tf.Summary()
            summary.value.add(tag='P@1', simple_value=p_at_1)
            summary.value.add(tag='MAP', simple_value=MAP_score)
            self.train_writer.add_summary(summary, self.g_step)
            self.train_writer.flush()

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
            prob = [(1,0.5), (2,0.2), (3,0.1), (4,0.1), (5,0.1)]
            #prob = [(1,0.9), (2,0.1)]
            v = random.random()
            for n, p in prob:
                v -= p
                if v < 0:
                    return n
            return 1


        target_class = ENLIDef.get_target_class(explain_tag)
        print(target_class)

        def save_payload(reinforce_payload, step_i):
            path_save = os.path.join(cpath.data_path, "nli_explain_train", "reinforce{}.pickle".format(step_i))
            pickle.dump(reinforce_payload, open(path_save, "wb"))

        def forward_runs(insts):
            alt_batches = get_batches_ex(insts, self.hparam.batch_size, 3)
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
            return alt_logits

        def train_explain(batch, step_i):
            summary = tf.Summary()

            ## Step 1) Prepare deletion RUNS
            def generate_alt_runs(batch):
                logits, ex_logit = self.sess.run([task.sout, task.conf_logits
                                                  ],
                                                 feed_dict=batch2feed_dict(batch)
                                                 )
                x0, x1, x2, y = batch

                pred = np.argmax(logits, axis=1)
                compare_deletion_num = 5
                instance_infos = []
                new_batches = []
                deleted_mask_list = []
                tag_size_list = []
                for i in range(len(logits)):
                    if pred[i] == target_class:
                        info = {}
                        info['init_logit'] = logits[i]
                        info['orig_input'] = (x0[i], x1[i], x2[i], y[i])
                        ex_tags = logit2tag(ex_logit[i])
                        self.log2.debug("EX_Score : {}".format(numpy_print(ex_logit[i])))
                        tag_size = np.count_nonzero(ex_tags)
                        tag_size_list.append(tag_size)
                        if tag_size > 10:
                            self.log2.debug("#Tagged token={}".format(tag_size))

                        info['idx_delete_tagged'] = len(new_batches)
                        new_batches.append(token_delete(ex_tags, x0[i], x1[i], x2[i]))
                        deleted_mask_list.append(ex_tags)

                        indice_delete_random = []

                        for _ in range(compare_deletion_num):
                            if multi_deletion:
                                tag_size = sample_size()
                            else:
                                tag_size = 1
                            indice_delete_random.append(len(new_batches))
                            x_list, delete_mask = random_delete(tag_size, x0[i], x1[i], x2[i])
                            new_batches.append(x_list)
                            deleted_mask_list.append(delete_mask)

                        info['indice_delete_random'] = indice_delete_random
                        instance_infos.append(info)
                if tag_size_list:
                    avg_tag_size = average(tag_size_list)
                    self.log2.debug("avg Tagged token#={}".format(avg_tag_size))
                return new_batches, instance_infos, deleted_mask_list

            new_batches, instance_infos, deleted_mask_list = generate_alt_runs(batch)

            if not new_batches:
                return
            ## Step 2) Execute deletion Runs
            alt_logits = forward_runs(new_batches)

            def reinforce_one(good_action, input_x):
                pos_reward_indice = np.int_(good_action)
                loss_mask = -pos_reward_indice + np.ones_like(pos_reward_indice) * 0.5
                x0,x1,x2,y = input_x
                reward_payload = (x0, x1, x2, y, loss_mask)
                return reward_payload

            def reinforce_binary(good_action, input_x):
                pos_reward_indice = np.int_(good_action)
                loss_mask = pos_reward_indice + np.ones_like(pos_reward_indice) * (-0.3)
                x0, x1, x2, y = input_x
                reward_payload = (x0, x1, x2, y, loss_mask)
                return reward_payload

            def supervise(good_action, input_x):
                label = np.int_(good_action)
                x0, x1, x2, y = input_x
                reward_payload = (x0, x1, x2, y, label)
                return reward_payload

            def reinforce_two(good_action, bad_action, input_x):
                pos_reward_indice = np.int_(np.logical_and(good_action, np.logical_not(bad_action)))
                neg_reward_indice = np.int_(np.logical_and(bad_action, np.logical_not(good_action)))
                loss_mask = -pos_reward_indice + neg_reward_indice
                x0,x1,x2,y = input_x

                self.log2.debug("Good: {}".format(good_action))
                self.log2.debug("Bad : {}".format(bad_action))
                self.log2.debug("Mask: {}".format(loss_mask))
                return (x0, x1, x2, y, loss_mask)

            if method in [0,1,4,5]:
                reinforce = reinforce_one
            elif method in [2]:
                reinforce = supervise
            elif method == 3:
                reinforce = reinforce_binary



            def action_score(before_prob, after_prob, action):
                num_tag = np.count_nonzero(action)
                if explain_tag == 'conflict':
                    penalty = (num_tag - 1) * 0.1 if num_tag > 1 else 0
                    score = (before_prob[2] - before_prob[0]) - (after_prob[2] - after_prob[0])
                elif explain_tag == 'match':
                    # Increase of neutral
                    penalty = (num_tag - 3) * 0.1 if num_tag > 3 else 0
                    score = (before_prob[2] + before_prob[0]) - (after_prob[2] + after_prob[0])
                     # ( 1 - before_prob[1] ) - (1 - after_prob[1]) = after_prob[1] - before_prob[1] = increase of neutral
                elif explain_tag == 'mismatch':
                    assert False
                    score = before_prob[1] - after_prob[1]
                elif explain_tag == 'dontcare':
                    assert False
                    score = sum([math.fabs(before_prob[i]-after_prob[i]) for i in [0,1,2]])
                else:
                    assert False
                score = score - penalty
                return score



            ## Step 3) Calc reward
            def calc_reward(alt_logits, instance_infos, deleted_mask_list):
                models_score_list = []
                reinforce_payload_list = []
                num_tag_list = []
                pos_win = 0
                pos_trial = 0
                for info in instance_infos:
                    init_output = info['init_logit']
                    models_after_output = alt_logits[info['idx_delete_tagged']]
                    input_x = info['orig_input']

                    predicted_action = deleted_mask_list[info['idx_delete_tagged']]
                    num_tag = np.count_nonzero(predicted_action)
                    num_tag_list.append(num_tag)
                    models_score = action_score(init_output, models_after_output, predicted_action)
                    models_score_list.append(models_score)
                    #self.log2.debug(
                    #    "target_ce_drop : {0:.4f}  n_token : {1}".format(target_ce_drop, num_tag))

                    good_action = predicted_action
                    best_score = models_score
                    for idx_delete_random in info['indice_delete_random']:
                        alt_after_output = alt_logits[idx_delete_random]
                        random_action = deleted_mask_list[idx_delete_random]
                        alt_score = action_score(init_output, alt_after_output, random_action)
                        if alt_score > best_score :
                            best_score = alt_score
                            good_action = random_action

                    reward_payload = reinforce(good_action, input_x)
                    reinforce_payload_list.append(reward_payload)
                    if models_score >= best_score:
                        pos_win += 1
                    pos_trial += 1

                match_rate = pos_win / pos_trial
                avg_score = average(models_score_list)
                self.log.debug("drop score : {0:.4f}  suc_rate : {1:0.2f}".format(avg_score, match_rate))
                summary.value.add(tag='#Tags', simple_value=average(num_tag_list))
                summary.value.add(tag='Score', simple_value=avg_score)
                summary.value.add(tag='Success', simple_value=match_rate)
                return reinforce_payload_list
            reinforce_payload = calc_reward(alt_logits, instance_infos, deleted_mask_list)

            def commit_reward(reinforce_payload):
                batches = get_batches_ex(reinforce_payload, self.hparam.batch_size, 5)
                rl_loss_list = []
                for batch in batches:
                    x0, x1, x2, y, rf_mask = batch
                    _, rl_loss, conf_logits,  = self.sess.run([train_rl, task.rl_loss,
                                                                                task.conf_logits,
                                                                                ],
                                            feed_dict={
                                                task.x_list[0]: x0,
                                                task.x_list[1]: x1,
                                                task.x_list[2]: x2,
                                                task.y : y,
                                                task.rf_mask : rf_mask,
                                            })
                    rl_loss_list.append(rl_loss)
                return average(rl_loss_list)

            save_payload(reinforce_payload, step_i)
            ## Step 4) Update gradient
            avg_rl_loss = commit_reward(reinforce_payload)

            summary.value.add(tag='RL_Loss', simple_value=avg_rl_loss)
            self.train_writer.add_summary(summary, self.g_step)
            self.train_writer.flush()


        def train_ex_fn(batch, step_i):
            loss_val, acc = train_classification(batch, step_i)
            train_explain(batch, step_i)
            self.g_step += 1
            return loss_val, acc

        def valid_fn():
            loss_list = []
            acc_list = []
            eval()
            for batch in dev_batches[:10]:
                loss_val, summary, acc = self.sess.run([task.loss, self.merged, task.acc],
                                                  feed_dict=batch2feed_dict(batch)
                                                  )
                loss_list.append(loss_val)
                acc_list.append(acc)

                self.test_writer.add_summary(summary, self.g_step)
            self.log.info("Validation : loss={0:.04f} acc={1:.04f}".format(average(loss_list), average(acc_list)))

        def save_fn():
            self.save_model(exp_config.name, 1)

        train_batches, dev_batches = self.load_nli_data(data_loader)
        valid_freq = 25
        steps = int(len(train_batches) * 0.5)
        loss, _ = sub_step_runner(train_batches, train_ex_fn,
                                  valid_fn, valid_freq,
                                  save_fn, self.save_interval,
                                  steps=steps)



    # Refactored version of explain train
    def train_nli_ex_with_premade_data(self, nli_setting, exp_config, data_loader, preload_id, explain_tag):
        enc_explain_dev, explain_dev = data_loader.get_dev_explain(explain_tag)
        method = 5
        task = transformer_nli(self.hparam, nli_setting.vocab_size, method)
        with tf.variable_scope("optimizer"):
            train_rl = self.get_train_op(task.rl_loss, name="rl")

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)

        if preload_id is not None:
            name = preload_id[0]
            id = preload_id[1]
            if exp_config.load_names:
                self.load_model_white(name, id, exp_config.load_names)
            else:
                assert False

        def eval(step_i):
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
            print(step_i)
            scores = eval_explain(conf_logit, data_loader, explain_tag)
            for key, value in scores.items():
                print(key, value)
            p_at_1, MAP_score = scores["P@1"], scores["MAP"]
            summary = tf.Summary()
            summary.value.add(tag='P@1', simple_value=p_at_1)
            summary.value.add(tag='MAP', simple_value=MAP_score)
            self.train_writer.add_summary(summary, self.g_step)
            self.train_writer.flush()

        target_class = ENLIDef.get_target_class(explain_tag)
        print(target_class)

        def train_explain(reinforce_payload, step_i):
            def commit_reward(reinforce_payload):
                batches = get_batches_ex(reinforce_payload, self.hparam.batch_size, 5)
                rl_loss_list = []
                for batch in batches:
                    x0, x1, x2, y, rf_mask = batch
                    _, rl_loss, conf_logits,  = self.sess.run([train_rl, task.rl_loss,
                                                                                task.conf_logits,
                                                                                ],
                                            feed_dict={
                                                task.x_list[0]: x0,
                                                task.x_list[1]: x1,
                                                task.x_list[2]: x2,
                                                task.y : y,
                                                task.rf_mask : rf_mask,
                                            })
                    rl_loss_list.append(rl_loss)
                return average(rl_loss_list)

            ## Step 4) Update gradient
            avg_rl_loss = commit_reward(reinforce_payload)


        def train_ex_fn(payload, step_i):
            loss_val, acc = 0, 0
            train_explain(payload, step_i)
            self.g_step += 1
            return loss_val, acc

        def valid_fn(step_i):
            eval(step_i)


        def save_fn():
            self.save_model(exp_config.name, 1)

        save_interval = 10 * 60
        valid_freq = 2000
        num_epochs = exp_config.num_epoch
        last_save = time.time()
        for i_epoch in range(1):
            step_size = 12000
            for step_i in range(step_size):
                if step_i % valid_freq == 0:
                    valid_fn(step_i)

                path_save = os.path.join(cpath.data_path, "nli_payload", explain_tag, "{}.pickle".format(step_i))
                if os.path.exists(path_save):
                    reinforce_payload = pickle.load(open(path_save, "rb"))
                    train_ex_fn(reinforce_payload, step_i)
                else:
                    print("Training data does not exists : ", path_save)
                    break

                if time.time() - last_save > save_interval:
                    save_fn()
                    last_save = time.time()
        save_fn()

    def train_nli_smart(self, nli_setting, exp_config, data_loader, preload_id, explain_tag, method):
        print("train_nli_smart")
        PAIRING_NLI = 8

        if exp_config.ex_val:
            enc_explain_dev, explain_dev = data_loader.get_dev_explain(explain_tag)
        if method == PAIRING_NLI:
            pair_dev = data_loader.get_pair_dev()
        else:
            pair_dev = None
        task = transformer_nli(self.hparam, nli_setting.vocab_size, method)
        with tf.variable_scope("optimizer"):
            train_cls = self.get_train_op(task.loss)
            train_rl = self.get_train_op(task.rl_loss, name="rl")

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)
        self.load_model_white2(preload_id, exp_config.load_names, False)
        if hasattr(exp_config, "save_eval"):
            eval_log = open(os.path.join(output_path, "report_log", exp_config.save_name), "w")


        def batch2feed_dict(batch):
            x0, x1, x2, y = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y: y,
            }
            return feed_dict

        def eval_tag():
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

            scores = eval_explain(conf_logit, data_loader, explain_tag)


            for metric in scores.keys():
                print("{}\t{}".format(metric, scores[metric]))

            if hasattr(exp_config, "save_eval"):
                eval_log.write("Eval\t{}\t{}\t{}\n".format(self.g_step, scores["P@1"], scores["MAP"]))
                eval_log.flush()
            p_at_1, MAP_score = scores["P@1"], scores["MAP"]
            summary = tf.Summary()
            summary.value.add(tag='P@1', simple_value=p_at_1)
            summary.value.add(tag='MAP', simple_value=MAP_score)
            self.train_writer.add_summary(summary, self.g_step)
            self.train_writer.flush()

        def eval_pair():
            p_enc, h_enc, pair_info = pair_dev

            def run(dev):
                batches = get_batches_ex(dev, self.hparam.batch_size, 3)
                pair_logit_list = []
                for batch in batches:
                    x0, x1, x2 = batch
                    logits, conf_logits = self.sess.run([task.sout, task.conf_logits],
                                                       feed_dict={
                                                           task.x_list[0]: x0,
                                                           task.x_list[1]: x1,
                                                           task.x_list[2]: x2,
                                                       })
                    pair_logit_list.append(conf_logits)
                pair_logits = np.concatenate(pair_logit_list)
                return pair_logits

            scores_p, scores_h = eval_pairing(run(p_enc), run(h_enc), data_loader, p_enc, pair_info)

            print("Pair for Prem")
            for metric in scores_p.keys():
                print("{}\t{}".format(metric, scores_p[metric]))

            print("Pair for Hypo")
            for metric in scores_h.keys():
                print("{}\t{}".format(metric, scores_h[metric]))

            summary = tf.Summary()
            summary.value.add(tag='P@1_pair_p', simple_value=scores_p["P@1"])
            summary.value.add(tag='AUC_pair_p', simple_value=scores_p["AUC"])
            summary.value.add(tag='P@1_pair_h', simple_value=scores_h["P@1"])
            summary.value.add(tag='AUC_pair_h', simple_value=scores_h["AUC"])
            self.train_writer.add_summary(summary, self.g_step)
            self.train_writer.flush()

        target_class_set = ENLIDef.get_target_class_set(explain_tag)
        print("target class : ", target_class_set)

        target_class = ENLIDef.get_target_class(explain_tag)
        logit2tag = over_zero

        def top_1_as_mask(np_arr):
            r = np.zeros_like(np_arr)
            r[np.argmax(np_arr)] = 1
            return r

        def forward_runs(insts):
            alt_batches = get_batches_ex(insts, self.hparam.batch_size, 3)
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
            return alt_logits

        def save_payload(reinforce_payload, step_i):
            path_save = os.path.join(cpath.data_path, "nli_payload", explain_tag,
                                     "{}.pickle".format(step_i))
            pickle.dump(reinforce_payload, open(path_save, "wb"))

        def sample_target_mask(batch):
            x0, x1, x2, y = batch

            def sample_it(segment_ids):
                num_mark = 1
                length = len(segment_ids)
                last_valid = 0
                for i in range(length):
                    if segment_ids[i] > 0:
                        last_valid = i

                def sample_len():
                    l = 1
                    v = random.random()
                    while v < 0.25 and l < length:
                        l = l * 2
                    return min(l, length)

                indice = []
                for i in range(num_mark):
                    mark_len = sample_len()
                    start_idx = pick1(range(last_valid + 1))
                    end_idx = min(start_idx + mark_len, last_valid + 1)

                    begin_segment_id = segment_ids[start_idx]
                    for idx in range(start_idx, end_idx):
                        if segment_ids[idx] != begin_segment_id:
                            break
                        indice.append(idx)

                for idx in indice:
                    segment_ids[idx] = ENLIDef.get_segment_marker(segment_ids[idx])

                return indice, segment_ids

            n_inst = len(batch)
            indice_list = []
            marked_inputs = []
            for i in range(n_inst):
                indice, x2_new = sample_it(x2[i])
                marked_inputs.append((x0[i], x1[i], x2_new, y[i]))
                indice_list.append(indice)

            mark_batches = get_batches_ex(marked_inputs, self.hparam.batch_size, 4)
            return indice_list, mark_batches[0]

        sample_deleter = get_seq_deleter(self.hparam.g_val)

        loss_window = MovingWindow(self.hparam.batch_size)
        def train_explain(batch, batch_info, step_i):
            summary = tf.Summary()
            def sample_size():
                prob = [(1,0.8), (2,0.2)]
                v = random.random()
                for n, p in prob:
                    v -= p
                    if v < 0:
                        return n
                return 1

            ## Step 1) Prepare deletion RUNS
            def generate_alt_runs(batch, batch_info):
                logits, ex_logit = self.sess.run([task.sout, task.conf_logits
                                                  ],
                                                 feed_dict=batch2feed_dict(batch)
                                                 )
                x0, x1, x2, y = batch


                pred = np.argmax(logits, axis=1)
                compare_deletion_num = self.hparam.compare_deletion_num
                instance_infos = []
                new_batches = []
                deleted_mask_list = []
                tag_size_list = []
                for i in range(len(logits)):
                    if pred[i] in target_class_set:
                        info = {}
                        info['init_logit'] = logits[i]
                        info['orig_input'] = (x0[i], x1[i], x2[i], y[i])
                        ex_tags = logit2tag(ex_logit[i])
                        self.log2.debug("EX_Score : {}".format(numpy_print(ex_logit[i])))
                        tag_size = np.count_nonzero(ex_tags)
                        tag_size_list.append(tag_size)
                        if tag_size > 10:
                            self.log2.debug("#Tagged token={}".format(tag_size))

                        info['idx_delete_tagged'] = len(new_batches)
                        new_batches.append(token_delete(ex_tags, x0[i], x1[i], x2[i]))
                        deleted_mask_list.append(ex_tags)

                        indice_delete_random = []

                        for _ in range(compare_deletion_num):
                            indice_delete_random.append(len(new_batches))
                            x_list, delete_mask = sample_deleter(sample_size(), x0[i], x1[i], x2[i])
                            new_batches.append(x_list)
                            deleted_mask_list.append(delete_mask)

                        info['indice_delete_random'] = indice_delete_random
                        instance_infos.append(info)
                if tag_size_list:
                    avg_tag_size = average(tag_size_list)
                    self.log2.debug("avg Tagged token#={}".format(avg_tag_size))
                return new_batches, instance_infos, deleted_mask_list

            new_batches, instance_infos, deleted_mask_list = generate_alt_runs(batch, batch_info)

            if not new_batches:
                return
            ## Step 2) Execute deletion Runs
            alt_logits = forward_runs(new_batches)

            def reinforce_one(good_action, input_x):
                pos_reward_indice = np.int_(good_action)
                loss_mask = -pos_reward_indice + np.ones_like(pos_reward_indice) * 0.1
                x0,x1,x2,y = input_x
                reward_payload = (x0, x1, x2, y, loss_mask)
                return reward_payload

            def reinforce_ce(good_action, input_x):
                pos_reward_indice = np.int_(good_action)
                loss_mask = pos_reward_indice
                x0,x1,x2,y = input_x
                reward_payload = (x0, x1, x2, y, loss_mask)
                return reward_payload

            reinforce = reinforce_one
            if method == 2 or method == 7:
                reinforce = reinforce_ce

            def action_score(before_prob, after_prob, action):
                num_tag = np.count_nonzero(action)
                penalty = (num_tag - 3) * 0.1 if num_tag > 3 else 0
                if explain_tag == 'conflict':
                    score = (before_prob[2] - before_prob[0]) - (after_prob[2] - after_prob[0])
                elif explain_tag == 'match':
                    # Increase of neutral
                    score = (before_prob[2] + before_prob[0]) - (after_prob[2] + after_prob[0])
                     # ( 1 - before_prob[1] ) - (1 - after_prob[1]) = after_prob[1] - before_prob[1] = increase of neutral
                elif explain_tag == 'mismatch':
                    score = before_prob[1] - after_prob[1]
                elif explain_tag == 'dontcare':
                    assert False
                    score = sum([math.fabs(before_prob[i]-after_prob[i]) for i in [0,1,2]])
                else:
                    assert False
                score = score - penalty
                return score

            ## Step 3) Calc reward
            def calc_reward(alt_logits, instance_infos, deleted_mask_list):
                models_score_list = []
                reinforce_payload_list = []
                num_tag_list = []
                pos_win = 0
                pos_trial = 0
                for info in instance_infos:
                    init_output = info['init_logit']
                    models_after_output = alt_logits[info['idx_delete_tagged']]
                    input_x = info['orig_input']

                    predicted_action = deleted_mask_list[info['idx_delete_tagged']]
                    num_tag = np.count_nonzero(predicted_action)
                    num_tag_list.append(num_tag)
                    models_score = action_score(init_output, models_after_output, predicted_action)
                    models_score_list.append(models_score)
                    #self.log2.debug(
                    #    "target_ce_drop : {0:.4f}  n_token : {1}".format(target_ce_drop, num_tag))

                    good_action = predicted_action
                    best_score = models_score
                    for idx_delete_random in info['indice_delete_random']:
                        alt_after_output = alt_logits[idx_delete_random]
                        random_action = deleted_mask_list[idx_delete_random]
                        alt_score = action_score(init_output, alt_after_output, random_action)
                        if alt_score > best_score :
                            best_score = alt_score
                            good_action = random_action

                    reward_payload = reinforce(good_action, input_x)
                    reinforce_payload_list.append(reward_payload)
                    if models_score >= best_score:
                        pos_win += 1
                    pos_trial += 1

                match_rate = pos_win / pos_trial
                avg_score = average(models_score_list)
                self.log.debug("drop score : {0:.4f}  suc_rate : {1:0.2f}".format(avg_score, match_rate))
                summary.value.add(tag='#Tags', simple_value=average(num_tag_list))
                summary.value.add(tag='Score', simple_value=avg_score)
                summary.value.add(tag='Success', simple_value=match_rate)
                return reinforce_payload_list

            reinforce_payload = calc_reward(alt_logits, instance_infos, deleted_mask_list)
            save_payload(reinforce_payload, step_i)

            def commit_reward(reinforce_payload):
                batches = get_batches_ex(reinforce_payload, self.hparam.batch_size, 5)
                rl_loss_list = []
                for batch in batches:
                    x0, x1, x2, y, rf_mask = batch
                    _, rl_loss, conf_logits,  = self.sess.run([train_rl, task.rl_loss,
                                                                                task.conf_logits,
                                                                                ],
                                            feed_dict={
                                                task.x_list[0]: x0,
                                                task.x_list[1]: x1,
                                                task.x_list[2]: x2,
                                                task.y : y,
                                                task.rf_mask : rf_mask,
                                            })
                    rl_loss_list.append(rl_loss)
                    loss_window.append(rl_loss, len(x0))
                return average(rl_loss_list)

            ## Step 4) Update gradient
            _ = commit_reward(reinforce_payload)

            window_rl_loss = loss_window.get_average()
            if hasattr(exp_config, "save_eval"):
                eval_log.write("Train_loss\t{}\t{}\n".format(self.g_step, window_rl_loss))

            summary.value.add(tag='RL_Loss', simple_value=window_rl_loss)
            self.train_writer.add_summary(summary, self.g_step)
            self.train_writer.flush()

        def train_pairing(batch_c, mark_loc, step_i):
            summary = tf.Summary()
            def sample_size():
                prob = [(1,0.8), (2,0.2)]
                v = random.random()
                for n, p in prob:
                    v -= p
                    if v < 0:
                        return n
                return 1

            def generate_alt_runs(batch, mark_loc):

                logits, ex_logit = self.sess.run([task.sout, task.conf_logits,
                                                  ],
                                                 feed_dict=batch2feed_dict(batch)
                                                 )
                x0, x1, x2, y = batch
                pred = np.argmax(logits, axis=1)
                compare_deletion_num = 20
                instance_infos = []
                new_batches = []
                deleted_mask_list = []
                for i in range(len(logits)):
                    info = {}
                    info['init_logits'] = logits[i]
                    info['pr_mark'] = mark_loc[i]
                    info['orig_input'] = (x0[i], x1[i], x2[i], y[i])
                    pr_tags = top_1_as_mask(ex_logit[i])
                    self.log2.debug("pr_scores : {}".format(numpy_print(ex_logit[i])))

                    info['idx_delete_tagged'] = len(new_batches)
                    new_batches.append(token_delete(pr_tags, x0[i], x1[i], x2[i]))
                    deleted_mask_list.append(pr_tags)
                    new_batches.append(seq_replace_inner(pr_tags, mark_loc[i], x0[i], x1[i], x2[i]))
                    deleted_mask_list.append(pr_tags)
                    indice_delete_random = []

                    for _ in range(compare_deletion_num):
                        #x_list, delete_mask = sample_deleter(sample_size(), None, data_loader.convert_indice_in,
                                                          #x0[i], x1[i], x2[i])
                        x_list_del, x_list_replace, replace_mask = seq_replace(sample_size(), mark_loc[i], x0[i], x1[i], x2[i])
                        indice_delete_random.append(len(new_batches))
                        new_batches.append(x_list_del)
                        deleted_mask_list.append(replace_mask)

                        indice_delete_random.append(len(new_batches))
                        new_batches.append(x_list_replace)
                        deleted_mask_list.append(replace_mask)


                    info['indice_delete_random'] = indice_delete_random
                    instance_infos.append(info)
                return new_batches, instance_infos, deleted_mask_list

            ## Step 1) Prepare deletion RUNS
            new_batches, instance_infos, deleted_mask_list = generate_alt_runs(batch_c, mark_loc)

            if not new_batches:
                return
            ## Step 2) Execute deletion Runs
            alt_logits = forward_runs(new_batches)

            def reinforce_one(good_action, input_x):
                pos_reward_indice = np.int_(good_action)
                loss_mask = -pos_reward_indice + np.ones_like(pos_reward_indice) * 0.5
                x0,x1,x2,y = input_x
                reward_payload = (x0, x1, x2, y, loss_mask)
                return reward_payload

            reinforce = reinforce_one

            s1_win = 0
            s2_win = 0
            def action_score_pr(pr_mark, before_logits, delete_logits, replace_logits, action):
                num_tag = np.count_nonzero(action)
                penalty = (num_tag - 4) * 0.1 if num_tag > 3 else 0

                s1 = replace_logits[0] - before_logits[0]
                s2 = replace_logits[0] - delete_logits[0]

                score = max(s1, s2)
                nonlocal s1_win, s2_win
                if s1 > s2 :
                    s1_win += 1
                elif s2 > s1:
                    s2_win += 1
                return score - penalty

            def select_best(alt_logits, instance_infos, deleted_mask_list):
                models_score_list = []
                reinforce_payload_list = []
                num_tag_list = []
                pos_win = 0
                pos_trial = 0
                for info in instance_infos:
                    init_logit = info['init_logits']
                    models_delete_logits = alt_logits[info['idx_delete_tagged']]
                    models_replace_logits = alt_logits[info['idx_delete_tagged']+1]
                    input_x = info['orig_input']
                    pr_mark = info['pr_mark'] # list[index]

                    predicted_action = deleted_mask_list[info['idx_delete_tagged']]
                    num_tag = np.count_nonzero(predicted_action)
                    num_tag_list.append(num_tag)
                    models_score = action_score_pr(pr_mark, init_logit, models_delete_logits, models_replace_logits, predicted_action)
                    models_score_list.append(models_score)
                    #self.log2.debug(
                    #    "target_ce_drop : {0:.4f}  n_token : {1}".format(target_ce_drop, num_tag))

                    good_action = predicted_action
                    best_score = models_score

                    alt_run_indice = info['indice_delete_random']
                    n_runs = len(alt_run_indice)
                    assert n_runs % 2 == 0
                    for i in range(0, n_runs, 2):
                        idx_delete_random = info['indice_delete_random'][i]
                        idx_replace_random = info['indice_delete_random'][i]
                        delete_logits = alt_logits[idx_delete_random]
                        replace_logits = alt_logits[idx_replace_random]
                        random_action = deleted_mask_list[idx_delete_random]
                        alt_score = action_score_pr(pr_mark, init_logit, delete_logits, replace_logits, random_action)
                        if alt_score > best_score :
                            best_score = alt_score
                            good_action = random_action

                    reward_payload = reinforce(good_action, input_x)
                    reinforce_payload_list.append(reward_payload)
                    if models_score >= best_score:
                        pos_win += 1
                    pos_trial += 1

                match_rate = pos_win / pos_trial
                avg_score = average(models_score_list)
                self.log.debug(" s1_win/s2_win = {}/{}".format(s1_win, s2_win))
                self.log.debug("[PR] drop score : {0:.4f}  suc_rate : {1:0.2f}".format(avg_score, match_rate))
                summary.value.add(tag='#Tags_PR', simple_value=average(num_tag_list))
                summary.value.add(tag='Score_PR', simple_value=avg_score)
                summary.value.add(tag='Success_PR', simple_value=match_rate)
                return reinforce_payload_list

            reinforce_payload = select_best(alt_logits, instance_infos, deleted_mask_list)


            def commit_reward(reinforce_payload):
                batches = get_batches_ex(reinforce_payload, self.hparam.batch_size, 5)
                rl_loss_list = []
                for batch in batches:
                    x0, x1, x2, y, loss_mask = batch
                    _, rl_loss, = self.sess.run([train_rl, task.rl_loss,
                                                                                ],
                                            feed_dict={
                                                task.x_list[0]: x0,
                                                task.x_list[1]: x1,
                                                task.x_list[2]: x2,
                                                task.y : y,
                                                task.rf_mask : loss_mask,
                                            })
                    rl_loss_list.append(rl_loss)
                return average(rl_loss_list)

            avg_rl_loss = commit_reward(reinforce_payload)

            summary.value.add(tag='PR_Loss', simple_value=avg_rl_loss)
            self.train_writer.add_summary(summary, self.g_step)
            self.train_writer.flush()


        def train_explain_with_addition_neutral(batch_c, step_i):
            summary = tf.Summary()
            def sample_size():
                prob = [(1,0.8), (2,0.2)]
                v = random.random()
                for n, p in prob:
                    v -= p
                    if v < 0:
                        return n
                return 1

            def generate_alt_runs(batch):
                logits, ex_logit = self.sess.run([task.sout, task.conf_logits,
                                                  ],
                                                 feed_dict=batch2feed_dict(batch)
                                                 )
                x0, x1, x2, y = batch
                pred = np.argmax(logits, axis=1)
                compare_deletion_num = 30
                instance_infos = []
                new_batches = []
                deleted_mask_list = []
                for i in range(len(logits)):
                    if pred[i] == target_class:
                        info = {}
                        info['init_logits'] = logits[i]
                        info['orig_input'] = (x0[i], x1[i], x2[i], y[i])
                        pr_tags = top_1_as_mask(ex_logit[i])
                        self.log2.debug("pr_scores : {}".format(numpy_print(ex_logit[i])))

                        indice_delete_random = []

                        for _ in range(compare_deletion_num):
                            #x_list, delete_mask = sample_deleter(sample_size(), None, data_loader.convert_indice_in,
                                                              #x0[i], x1[i], x2[
                            x_list_del, delete_mask = sample_deleter(sample_size(),
                                                              x0[i], x1[i], x2[i])

                            indice_delete_random.append(len(new_batches))
                            new_batches.append(x_list_del)
                            deleted_mask_list.append(delete_mask)

                            x_list_add, src_mask = add_seq_hypo2prem(x0[i], x1[i], x2[i])
                            indice_delete_random.append(len(new_batches))
                            new_batches.append(x_list_add)
                            deleted_mask_list.append(src_mask)

                        info['indice_delete_random'] = indice_delete_random
                        instance_infos.append(info)
                return new_batches, instance_infos, deleted_mask_list

            ## Step 1) Prepare deletion RUNS
            new_batches, instance_infos, deleted_mask_list = generate_alt_runs(batch_c)

            if not new_batches:
                return
            ## Step 2) Execute deletion Runs
            alt_logits = forward_runs(new_batches)

            def reinforce_one(good_action, input_x):
                pos_reward_indice = np.int_(good_action)
                loss_mask = -pos_reward_indice + np.ones_like(pos_reward_indice) * 0.5
                x0,x1,x2,y = input_x
                reward_payload = (x0, x1, x2, y, loss_mask)
                return reward_payload

            reinforce = reinforce_one

            s1_win = 0
            s2_win = 0
            def action_score(before_logits, alt_logits, action):
                num_tag = np.count_nonzero(action)
                penalty = (num_tag - 4) * 0.07 if num_tag > 3 else 0
                score = (before_logits[1] - alt_logits[1])
                return score - penalty

            def select_best(alt_logits, instance_infos, deleted_mask_list):
                models_score_list = []
                reinforce_payload_list = []
                num_tag_list = []
                for info in instance_infos:
                    init_logit = info['init_logits']
                    input_x = info['orig_input']

                    good_action = None
                    best_score = -999

                    alt_run_indice = info['indice_delete_random']
                    n_runs = len(alt_run_indice)
                    assert n_runs % 2 == 0
                    best_i = 0
                    for i in range(0, n_runs):
                        idx_alt = info['indice_delete_random'][i]
                        alt_logit = alt_logits[idx_alt]
                        alt_action = deleted_mask_list[idx_alt]

                        alt_score = action_score(init_logit, alt_logit, alt_action)
                        if alt_score > best_score :
                            best_score = alt_score
                            good_action = alt_action
                            best_i = i

                    nonlocal s1_win, s2_win
                    if best_i % 2 == 0:
                        s1_win += 1
                    else:
                        s2_win += 1
                    reward_payload = reinforce(good_action, input_x)
                    reinforce_payload_list.append(reward_payload)

                avg_score = average(models_score_list)
                self.log.debug(" s1_win/s2_win = {}/{}".format(s1_win, s2_win))
                self.log.debug("[PR] drop score : {0:.4f}".format(avg_score))
                summary.value.add(tag='#Tags_PR', simple_value=average(num_tag_list))
                summary.value.add(tag='Score_PR', simple_value=avg_score)
                return reinforce_payload_list

            reinforce_payload = select_best(alt_logits, instance_infos, deleted_mask_list)


            def commit_reward(reinforce_payload):
                batches = get_batches_ex(reinforce_payload, self.hparam.batch_size, 5)
                rl_loss_list = []
                for batch in batches:
                    x0, x1, x2, y, loss_mask = batch
                    _, rl_loss, = self.sess.run([train_rl, task.rl_loss,
                                                                                ],
                                            feed_dict={
                                                task.x_list[0]: x0,
                                                task.x_list[1]: x1,
                                                task.x_list[2]: x2,
                                                task.y : y,
                                                task.rf_mask : loss_mask,
                                            })
                    rl_loss_list.append(rl_loss)
                return average(rl_loss_list)

            avg_rl_loss = commit_reward(reinforce_payload)

            summary.value.add(tag='PR_Loss', simple_value=avg_rl_loss)
            self.train_writer.add_summary(summary, self.g_step)
            self.train_writer.flush()



        def train_classification(batch, step_i):
            loss_val, summary, acc,  _ = self.sess.run([task.loss, self.merged, task.acc, train_cls,
                                             ],
                                            feed_dict=batch2feed_dict(batch)
                                            )
            self.log.debug("Step {0} train loss={1:.04f} acc={2:.03f}".format(step_i, loss_val, acc))
            self.train_writer.add_summary(summary, self.g_step)
            return loss_val, acc


        def train_fn(batch, step_i):
            batch_c, batch_info = batch
            if method == PAIRING_NLI:
                mark_loc, mark_batch = sample_target_mask(batch_c)
                loss_val, acc = train_classification(mark_batch, step_i)
                train_pairing(mark_batch, mark_loc, step_i)
            #elif explain_tag == 'mismatch':
            #    loss_val, acc = train_classification(batch_c, step_i)
            #    train_explain_with_addition_neutral(batch_c, step_i)
            else:
                loss_val, acc = train_classification(batch_c, step_i)
                train_explain(batch_c, batch_info, step_i)
            self.g_step += 1
            return loss_val, acc


        def eval_acc():
            loss_list = []
            acc_list = []
            for batch in dev_batches[:10]:
                loss_val, summary, acc = self.sess.run([task.loss, self.merged, task.acc],
                                                  feed_dict=batch2feed_dict(batch)
                                                  )
                loss_list.append(loss_val)
                acc_list.append(acc)

                self.test_writer.add_summary(summary, self.g_step)
            self.log.info("Validation : loss={0:.04f} acc={1:.04f}".format(average(loss_list), average(acc_list)))


        def valid_fn():
            eval_acc()
            if exp_config.ex_val:
                if method == PAIRING_NLI:
                    eval_pair()
                else:
                    eval_tag()

        def save_fn():
            self.save_model(exp_config.name, 1)

        train_batches, train_batches_info, dev_batches = self.load_nli_data_with_info(data_loader)
        train_batches = list(zip(train_batches, train_batches_info))

        valid_freq = 1000
        print("Total of {} train batches".format(len(train_batches)))
        steps = int(len(train_batches) * 0.5)
        loss, _ = sub_step_runner(train_batches, train_fn,
                                  valid_fn, valid_freq,
                                  save_fn, self.save_interval,
                                  steps=steps)


    def train_nli_any_way(self, nli_setting, exp_config, data_loader, preload_id):
        print("train_nli_any_way")

        task = transformer_nli(self.hparam, nli_setting.vocab_size, 5)
        with tf.variable_scope("optimizer"):
            train_cls = self.get_train_op(task.loss)
            train_rl = self.get_train_op(task.rl_loss, name="rl")

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)

        if exp_config.v2_load:
            self.load_model_v2_map(preload_id, exp_config.load_names)
        else:
            self.load_model_white2(preload_id, exp_config.load_names)

        train_batches, dev_batches = self.load_nli_data(data_loader)

        def batch2feed_dict(batch):
            x0, x1, x2, y = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y: y,
            }
            return feed_dict



        logit2tag = over_zero

        def top_1_as_mask(np_arr):
            r = np.zeros_like(np_arr)
            r[np.argmax(np_arr)] = 1
            return r

        def forward_runs(insts):
            alt_batches = get_batches_ex(insts, self.hparam.batch_size, 3)
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
            return alt_logits

        loss_window = MovingWindow(self.hparam.batch_size)
        def train_explain(batch, step_i):
            summary = tf.Summary()

            def sample_size():
                prob = [(1,0.8), (2,0.2)]
                v = random.random()
                for n, p in prob:
                    v -= p
                    if v < 0:
                        return n
                return 1

            ## Step 1) Prepare deletion RUNS
            def generate_alt_runs(batch):
                logits, ex_logit = self.sess.run([task.sout, task.conf_logits
                                                  ],
                                                 feed_dict=batch2feed_dict(batch)
                                                 )
                x0, x1, x2, y = batch


                pred = np.argmax(logits, axis=1)
                compare_deletion_num = self.hparam.compare_deletion_num
                instance_infos = []
                new_batches = []
                deleted_mask_list = []
                tag_size_list = []
                for i in range(len(logits)):
                    info = {}
                    info['init_logit'] = logits[i]
                    info['orig_input'] = (x0[i], x1[i], x2[i], y[i])
                    ex_tags = logit2tag(ex_logit[i])
                    self.log2.debug("EX_Score : {}".format(numpy_print(ex_logit[i])))
                    tag_size = np.count_nonzero(ex_tags)
                    tag_size_list.append(tag_size)
                    if tag_size > 10:
                        self.log2.debug("#Tagged token={}".format(tag_size))

                    info['idx_delete_tagged'] = len(new_batches)
                    new_batches.append(token_delete(ex_tags, x0[i], x1[i], x2[i]))
                    deleted_mask_list.append(ex_tags)

                    indice_delete_random = []

                    for _ in range(compare_deletion_num):
                        indice_delete_random.append(len(new_batches))
                        x_list, delete_mask = seq_delete_inner(sample_size(), x0[i], x1[i], x2[i])
                        new_batches.append(x_list)
                        deleted_mask_list.append(delete_mask)

                    info['indice_delete_random'] = indice_delete_random
                    instance_infos.append(info)
                if tag_size_list:
                    avg_tag_size = average(tag_size_list)
                    self.log2.debug("avg Tagged token#={}".format(avg_tag_size))
                return new_batches, instance_infos, deleted_mask_list

            new_batches, instance_infos, deleted_mask_list = generate_alt_runs(batch)

            if not new_batches:
                return
            ## Step 2) Execute deletion Runs
            alt_logits = forward_runs(new_batches)

            def reinforce_one(good_action, input_x):
                pos_reward_indice = np.int_(good_action)
                loss_mask = -pos_reward_indice + np.ones_like(pos_reward_indice) * 0.1
                x0,x1,x2,y = input_x
                reward_payload = (x0, x1, x2, y, loss_mask)
                return reward_payload

            reinforce = reinforce_one

            def action_score(before_prob, after_prob, action):
                num_tag = np.count_nonzero(action)
                orig_pred = np.argmax(before_prob)
                penalty = (num_tag - 3) * 0.1 if num_tag > 3 else 0
                score = before_prob[orig_pred] - after_prob[orig_pred]
                score = score - penalty
                return score

            ## Step 3) Calc reward
            def calc_reward(alt_logits, instance_infos, deleted_mask_list):
                models_score_list = []
                reinforce_payload_list = []
                num_tag_list = []
                pos_win = 0
                pos_trial = 0
                for info in instance_infos:
                    init_output = info['init_logit']
                    models_after_output = alt_logits[info['idx_delete_tagged']]
                    input_x = info['orig_input']

                    predicted_action = deleted_mask_list[info['idx_delete_tagged']]
                    num_tag = np.count_nonzero(predicted_action)
                    num_tag_list.append(num_tag)
                    models_score = action_score(init_output, models_after_output, predicted_action)
                    models_score_list.append(models_score)
                    #self.log2.debug(
                    #    "target_ce_drop : {0:.4f}  n_token : {1}".format(target_ce_drop, num_tag))

                    good_action = predicted_action
                    best_score = models_score
                    for idx_delete_random in info['indice_delete_random']:
                        alt_after_output = alt_logits[idx_delete_random]
                        random_action = deleted_mask_list[idx_delete_random]
                        alt_score = action_score(init_output, alt_after_output, random_action)
                        if alt_score > best_score :
                            best_score = alt_score
                            good_action = random_action

                    reward_payload = reinforce(good_action, input_x)
                    reinforce_payload_list.append(reward_payload)
                    if models_score >= best_score:
                        pos_win += 1
                    pos_trial += 1

                match_rate = pos_win / pos_trial
                avg_score = average(models_score_list)
                self.log.debug("drop score : {0:.4f}  suc_rate : {1:0.2f}".format(avg_score, match_rate))
                summary.value.add(tag='#Tags', simple_value=average(num_tag_list))
                summary.value.add(tag='Score', simple_value=avg_score)
                summary.value.add(tag='Success', simple_value=match_rate)
                return reinforce_payload_list

            reinforce_payload = calc_reward(alt_logits, instance_infos, deleted_mask_list)

            def commit_reward(reinforce_payload):
                batches = get_batches_ex(reinforce_payload, self.hparam.batch_size, 5)
                rl_loss_list = []
                for batch in batches:
                    x0, x1, x2, y, rf_mask = batch
                    _, rl_loss, conf_logits,  = self.sess.run([train_rl, task.rl_loss,
                                                                                task.conf_logits,
                                                                                ],
                                            feed_dict={
                                                task.x_list[0]: x0,
                                                task.x_list[1]: x1,
                                                task.x_list[2]: x2,
                                                task.y : y,
                                                task.rf_mask : rf_mask,
                                            })
                    rl_loss_list.append(rl_loss)
                    loss_window.append(rl_loss, len(x0))
                return average(rl_loss_list)

            ## Step 4) Update gradient
            _ = commit_reward(reinforce_payload)

            window_rl_loss = loss_window.get_average()
            summary.value.add(tag='RL_Loss', simple_value=window_rl_loss)
            self.train_writer.add_summary(summary, self.g_step)
            self.train_writer.flush()

        def train_pairing(batch_c, mark_loc, step_i):
            summary = tf.Summary()
            def sample_size():
                prob = [(1,0.8), (2,0.2)]
                v = random.random()
                for n, p in prob:
                    v -= p
                    if v < 0:
                        return n
                return 1

            def generate_alt_runs(batch, mark_loc):

                logits, ex_logit = self.sess.run([task.sout, task.conf_logits,
                                                  ],
                                                 feed_dict=batch2feed_dict(batch)
                                                 )
                x0, x1, x2, y = batch
                pred = np.argmax(logits, axis=1)
                compare_deletion_num = 20
                instance_infos = []
                new_batches = []
                deleted_mask_list = []
                for i in range(len(logits)):
                    info = {}
                    info['init_logits'] = logits[i]
                    info['pr_mark'] = mark_loc[i]
                    info['orig_input'] = (x0[i], x1[i], x2[i], y[i])
                    pr_tags = top_1_as_mask(ex_logit[i])
                    self.log2.debug("pr_scores : {}".format(numpy_print(ex_logit[i])))

                    info['idx_delete_tagged'] = len(new_batches)
                    new_batches.append(token_delete(pr_tags, x0[i], x1[i], x2[i]))
                    deleted_mask_list.append(pr_tags)
                    new_batches.append(seq_replace_inner(pr_tags, mark_loc[i], x0[i], x1[i], x2[i]))
                    deleted_mask_list.append(pr_tags)
                    indice_delete_random = []

                    for _ in range(compare_deletion_num):
                        #x_list, delete_mask = sample_deleter(sample_size(), None, data_loader.convert_indice_in,
                                                          #x0[i], x1[i], x2[i])
                        x_list_del, x_list_replace, replace_mask = seq_replace(sample_size(), mark_loc[i], x0[i], x1[i], x2[i])
                        indice_delete_random.append(len(new_batches))
                        new_batches.append(x_list_del)
                        deleted_mask_list.append(replace_mask)

                        indice_delete_random.append(len(new_batches))
                        new_batches.append(x_list_replace)
                        deleted_mask_list.append(replace_mask)


                    info['indice_delete_random'] = indice_delete_random
                    instance_infos.append(info)
                return new_batches, instance_infos, deleted_mask_list

            ## Step 1) Prepare deletion RUNS
            new_batches, instance_infos, deleted_mask_list = generate_alt_runs(batch_c, mark_loc)

            if not new_batches:
                return
            ## Step 2) Execute deletion Runs
            alt_logits = forward_runs(new_batches)

            def reinforce_one(good_action, input_x):
                pos_reward_indice = np.int_(good_action)
                loss_mask = -pos_reward_indice + np.ones_like(pos_reward_indice) * 0.5
                x0,x1,x2,y = input_x
                reward_payload = (x0, x1, x2, y, loss_mask)
                return reward_payload

            reinforce = reinforce_one

            s1_win = 0
            s2_win = 0
            def action_score_pr(pr_mark, before_logits, delete_logits, replace_logits, action):
                num_tag = np.count_nonzero(action)
                penalty = (num_tag - 4) * 0.1 if num_tag > 3 else 0

                s1 = replace_logits[0] - before_logits[0]
                s2 = replace_logits[0] - delete_logits[0]

                score = max(s1, s2)
                nonlocal s1_win, s2_win
                if s1 > s2 :
                    s1_win += 1
                elif s2 > s1:
                    s2_win += 1
                return score - penalty

            def select_best(alt_logits, instance_infos, deleted_mask_list):
                models_score_list = []
                reinforce_payload_list = []
                num_tag_list = []
                pos_win = 0
                pos_trial = 0
                for info in instance_infos:
                    init_logit = info['init_logits']
                    models_delete_logits = alt_logits[info['idx_delete_tagged']]
                    models_replace_logits = alt_logits[info['idx_delete_tagged']+1]
                    input_x = info['orig_input']
                    pr_mark = info['pr_mark'] # list[index]

                    predicted_action = deleted_mask_list[info['idx_delete_tagged']]
                    num_tag = np.count_nonzero(predicted_action)
                    num_tag_list.append(num_tag)
                    models_score = action_score_pr(pr_mark, init_logit, models_delete_logits, models_replace_logits, predicted_action)
                    models_score_list.append(models_score)
                    #self.log2.debug(
                    #    "target_ce_drop : {0:.4f}  n_token : {1}".format(target_ce_drop, num_tag))

                    good_action = predicted_action
                    best_score = models_score

                    alt_run_indice = info['indice_delete_random']
                    n_runs = len(alt_run_indice)
                    assert n_runs % 2 == 0
                    for i in range(0, n_runs, 2):
                        idx_delete_random = info['indice_delete_random'][i]
                        idx_replace_random = info['indice_delete_random'][i]
                        delete_logits = alt_logits[idx_delete_random]
                        replace_logits = alt_logits[idx_replace_random]
                        random_action = deleted_mask_list[idx_delete_random]
                        alt_score = action_score_pr(pr_mark, init_logit, delete_logits, replace_logits, random_action)
                        if alt_score > best_score :
                            best_score = alt_score
                            good_action = random_action

                    reward_payload = reinforce(good_action, input_x)
                    reinforce_payload_list.append(reward_payload)
                    if models_score >= best_score:
                        pos_win += 1
                    pos_trial += 1

                match_rate = pos_win / pos_trial
                avg_score = average(models_score_list)
                self.log.debug(" s1_win/s2_win = {}/{}".format(s1_win, s2_win))
                self.log.debug("[PR] drop score : {0:.4f}  suc_rate : {1:0.2f}".format(avg_score, match_rate))
                summary.value.add(tag='#Tags_PR', simple_value=average(num_tag_list))
                summary.value.add(tag='Score_PR', simple_value=avg_score)
                summary.value.add(tag='Success_PR', simple_value=match_rate)
                return reinforce_payload_list

            reinforce_payload = select_best(alt_logits, instance_infos, deleted_mask_list)


            def commit_reward(reinforce_payload):
                batches = get_batches_ex(reinforce_payload, self.hparam.batch_size, 5)
                rl_loss_list = []
                for batch in batches:
                    x0, x1, x2, y, loss_mask = batch
                    _, rl_loss, = self.sess.run([train_rl, task.rl_loss,
                                                                                ],
                                            feed_dict={
                                                task.x_list[0]: x0,
                                                task.x_list[1]: x1,
                                                task.x_list[2]: x2,
                                                task.y : y,
                                                task.rf_mask : loss_mask,
                                            })
                    rl_loss_list.append(rl_loss)
                return average(rl_loss_list)

            avg_rl_loss = commit_reward(reinforce_payload)

            summary.value.add(tag='PR_Loss', simple_value=avg_rl_loss)
            self.train_writer.add_summary(summary, self.g_step)
            self.train_writer.flush()

        def train_classification(batch, step_i):
            loss_val, summary, acc,  _ = self.sess.run([task.loss, self.merged, task.acc, train_cls,
                                             ],
                                            feed_dict=batch2feed_dict(batch)
                                            )
            self.log.debug("Step {0} train loss={1:.04f} acc={2:.03f}".format(step_i, loss_val, acc))
            self.train_writer.add_summary(summary, self.g_step)
            return loss_val, acc


        def train_fn(batch, step_i):
            loss_val, acc = train_classification(batch, step_i)
            train_explain(batch, step_i)
            self.g_step += 1
            return loss_val, acc


        def eval_acc():
            loss_list = []
            acc_list = []
            for batch in dev_batches[:10]:
                loss_val, summary, acc = self.sess.run([task.loss, self.merged, task.acc],
                                                  feed_dict=batch2feed_dict(batch)
                                                  )
                loss_list.append(loss_val)
                acc_list.append(acc)

                self.test_writer.add_summary(summary, self.g_step)
            self.log.info("Validation : loss={0:.04f} acc={1:.04f}".format(average(loss_list), average(acc_list)))

        def valid_fn():
            eval_acc()

        def save_fn():
            self.save_model(exp_config.name, 1)


        valid_freq = 25
        print("Total of {} train batches".format(len(train_batches)))
        steps = int(len(train_batches) * 0.3)
        loss, _ = sub_step_runner(train_batches, train_fn,
                                  valid_fn, valid_freq,
                                  save_fn, self.save_interval,
                                  steps=steps)


    def train_nli_v2(self, nli_setting, exp_config, data_loader, preload_id, explain_tag):
        print("train_nli_v2")

        if exp_config.ex_val:
            enc_explain_dev, explain_dev = data_loader.get_dev_explain(explain_tag)
        task = transformer_nli(self.hparam, nli_setting.vocab_size, 5)
        with tf.variable_scope("optimizer"):
            train_cls = self.get_train_op(task.loss)
            train_rl = self.get_train_op(task.rl_loss, name="rl")

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)
        self.load_model_white2(preload_id, exp_config.load_names)

        train_batches, dev_batches = self.load_nli_data(data_loader)

        def batch2feed_dict(batch):
            x0, x1, x2, y = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y: y,
            }
            return feed_dict

        def eval_tag():
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

            scores = eval_explain(conf_logit, data_loader, explain_tag)
            for metric in scores.keys():
                print("{}\t{}".format(metric, scores[metric]))

            p_at_1, MAP_score = scores["P@1"], scores["MAP"]
            summary = tf.Summary()
            summary.value.add(tag='P@1', simple_value=p_at_1)
            summary.value.add(tag='MAP', simple_value=MAP_score)
            self.train_writer.add_summary(summary, self.g_step)
            self.train_writer.flush()

        target_class_set = ENLIDef.get_target_class_set(explain_tag)
        print("target class : ", target_class_set)


        logit2tag = over_zero

        def top_1_as_mask(np_arr):
            r = np.zeros_like(np_arr)
            r[np.argmax(np_arr)] = 1
            return r

        def forward_runs(insts):
            alt_batches = get_batches_ex(insts, self.hparam.batch_size, 3)
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
            return alt_logits

        loss_window = MovingWindow(self.hparam.batch_size)
        def train_explain(batch, step_i):
            summary = tf.Summary()

            def sample_size():
                prob = [(1,0.8), (2,0.2)]
                v = random.random()
                for n, p in prob:
                    v -= p
                    if v < 0:
                        return n
                return 1

            ## Step 1) Prepare deletion RUNS
            def generate_alt_runs(batch):
                logits, ex_logit = self.sess.run([task.sout, task.conf_logits
                                                  ],
                                                 feed_dict=batch2feed_dict(batch)
                                                 )
                x0, x1, x2, y = batch


                pred = np.argmax(logits, axis=1)
                compare_deletion_num = self.hparam.compare_deletion_num
                instance_infos = []
                new_batches = []
                deleted_mask_list = []
                tag_size_list = []
                for i in range(len(logits)):
                    if pred[i] in target_class_set:
                        info = {}
                        info['init_logit'] = logits[i]
                        info['orig_input'] = (x0[i], x1[i], x2[i], y[i])
                        ex_tags = logit2tag(ex_logit[i])
                        self.log2.debug("EX_Score : {}".format(numpy_print(ex_logit[i])))
                        tag_size = np.count_nonzero(ex_tags)
                        tag_size_list.append(tag_size)
                        if tag_size > 10:
                            self.log2.debug("#Tagged token={}".format(tag_size))

                        info['idx_delete_tagged'] = len(new_batches)
                        new_batches.append(token_delete(ex_tags, x0[i], x1[i], x2[i]))
                        deleted_mask_list.append(ex_tags)

                        indice_delete_random = []

                        for _ in range(compare_deletion_num):
                            indice_delete_random.append(len(new_batches))
                            x_list, delete_mask = seq_delete_inner(sample_size(), x0[i], x1[i], x2[i])
                            new_batches.append(x_list)
                            deleted_mask_list.append(delete_mask)

                        info['indice_delete_random'] = indice_delete_random
                        instance_infos.append(info)
                if tag_size_list:
                    avg_tag_size = average(tag_size_list)
                    self.log2.debug("avg Tagged token#={}".format(avg_tag_size))
                return new_batches, instance_infos, deleted_mask_list

            new_batches, instance_infos, deleted_mask_list = generate_alt_runs(batch)

            if not new_batches:
                return
            ## Step 2) Execute deletion Runs
            alt_logits = forward_runs(new_batches)

            def reinforce_one(good_action, input_x):
                pos_reward_indice = np.int_(good_action)
                loss_mask = -pos_reward_indice + np.ones_like(pos_reward_indice) * 0.1
                x0,x1,x2,y = input_x
                reward_payload = (x0, x1, x2, y, loss_mask)
                return reward_payload

            reinforce = reinforce_one

            def action_score(before_prob, after_prob, action):
                num_tag = np.count_nonzero(action)
                penalty = (num_tag - 3) * 0.1 if num_tag > 3 else 0
                if explain_tag == 'conflict':
                    score = (before_prob[2] - before_prob[0]) - (after_prob[2] - after_prob[0])
                elif explain_tag == 'match':
                    # Increase of neutral
                    score = (before_prob[2] + before_prob[0]) - (after_prob[2] + after_prob[0])
                     # ( 1 - before_prob[1] ) - (1 - after_prob[1]) = after_prob[1] - before_prob[1] = increase of neutral
                elif explain_tag == 'mismatch':
                    score = before_prob[1] - after_prob[1]
                else:
                    assert False
                score = score - penalty
                return score

            ## Step 3) Calc reward
            def calc_reward(alt_logits, instance_infos, deleted_mask_list):
                models_score_list = []
                reinforce_payload_list = []
                num_tag_list = []
                pos_win = 0
                pos_trial = 0
                for info in instance_infos:
                    init_output = info['init_logit']
                    models_after_output = alt_logits[info['idx_delete_tagged']]
                    input_x = info['orig_input']

                    predicted_action = deleted_mask_list[info['idx_delete_tagged']]
                    num_tag = np.count_nonzero(predicted_action)
                    num_tag_list.append(num_tag)
                    models_score = action_score(init_output, models_after_output, predicted_action)
                    models_score_list.append(models_score)
                    #self.log2.debug(
                    #    "target_ce_drop : {0:.4f}  n_token : {1}".format(target_ce_drop, num_tag))

                    good_action = predicted_action
                    best_score = models_score
                    for idx_delete_random in info['indice_delete_random']:
                        alt_after_output = alt_logits[idx_delete_random]
                        random_action = deleted_mask_list[idx_delete_random]
                        alt_score = action_score(init_output, alt_after_output, random_action)
                        if alt_score > best_score :
                            best_score = alt_score
                            good_action = random_action

                    reward_payload = reinforce(good_action, input_x)
                    reinforce_payload_list.append(reward_payload)
                    if models_score >= best_score:
                        pos_win += 1
                    pos_trial += 1

                match_rate = pos_win / pos_trial
                avg_score = average(models_score_list)
                self.log.debug("drop score : {0:.4f}  suc_rate : {1:0.2f}".format(avg_score, match_rate))
                summary.value.add(tag='#Tags', simple_value=average(num_tag_list))
                summary.value.add(tag='Score', simple_value=avg_score)
                summary.value.add(tag='Success', simple_value=match_rate)
                return reinforce_payload_list

            reinforce_payload = calc_reward(alt_logits, instance_infos, deleted_mask_list)

            def commit_reward(reinforce_payload):
                batches = get_batches_ex(reinforce_payload, self.hparam.batch_size, 5)
                rl_loss_list = []
                for batch in batches:
                    x0, x1, x2, y, rf_mask = batch
                    _, rl_loss, conf_logits,  = self.sess.run([train_rl, task.rl_loss,
                                                                                task.conf_logits,
                                                                                ],
                                            feed_dict={
                                                task.x_list[0]: x0,
                                                task.x_list[1]: x1,
                                                task.x_list[2]: x2,
                                                task.y : y,
                                                task.rf_mask : rf_mask,
                                            })
                    rl_loss_list.append(rl_loss)
                    loss_window.append(rl_loss, len(x0))
                return average(rl_loss_list)

            ## Step 4) Update gradient
            _ = commit_reward(reinforce_payload)

            window_rl_loss = loss_window.get_average()
            summary.value.add(tag='RL_Loss', simple_value=window_rl_loss)
            self.train_writer.add_summary(summary, self.g_step)
            self.train_writer.flush()

        def train_pairing(batch_c, mark_loc, step_i):
            summary = tf.Summary()
            def sample_size():
                prob = [(1,0.8), (2,0.2)]
                v = random.random()
                for n, p in prob:
                    v -= p
                    if v < 0:
                        return n
                return 1

            def generate_alt_runs(batch, mark_loc):

                logits, ex_logit = self.sess.run([task.sout, task.conf_logits,
                                                  ],
                                                 feed_dict=batch2feed_dict(batch)
                                                 )
                x0, x1, x2, y = batch
                pred = np.argmax(logits, axis=1)
                compare_deletion_num = 20
                instance_infos = []
                new_batches = []
                deleted_mask_list = []
                for i in range(len(logits)):
                    info = {}
                    info['init_logits'] = logits[i]
                    info['pr_mark'] = mark_loc[i]
                    info['orig_input'] = (x0[i], x1[i], x2[i], y[i])
                    pr_tags = top_1_as_mask(ex_logit[i])
                    self.log2.debug("pr_scores : {}".format(numpy_print(ex_logit[i])))

                    info['idx_delete_tagged'] = len(new_batches)
                    new_batches.append(token_delete(pr_tags, x0[i], x1[i], x2[i]))
                    deleted_mask_list.append(pr_tags)
                    new_batches.append(seq_replace_inner(pr_tags, mark_loc[i], x0[i], x1[i], x2[i]))
                    deleted_mask_list.append(pr_tags)
                    indice_delete_random = []

                    for _ in range(compare_deletion_num):
                        #x_list, delete_mask = sample_deleter(sample_size(), None, data_loader.convert_indice_in,
                                                          #x0[i], x1[i], x2[i])
                        x_list_del, x_list_replace, replace_mask = seq_replace(sample_size(), mark_loc[i], x0[i], x1[i], x2[i])
                        indice_delete_random.append(len(new_batches))
                        new_batches.append(x_list_del)
                        deleted_mask_list.append(replace_mask)

                        indice_delete_random.append(len(new_batches))
                        new_batches.append(x_list_replace)
                        deleted_mask_list.append(replace_mask)


                    info['indice_delete_random'] = indice_delete_random
                    instance_infos.append(info)
                return new_batches, instance_infos, deleted_mask_list

            ## Step 1) Prepare deletion RUNS
            new_batches, instance_infos, deleted_mask_list = generate_alt_runs(batch_c, mark_loc)

            if not new_batches:
                return
            ## Step 2) Execute deletion Runs
            alt_logits = forward_runs(new_batches)

            def reinforce_one(good_action, input_x):
                pos_reward_indice = np.int_(good_action)
                loss_mask = -pos_reward_indice + np.ones_like(pos_reward_indice) * 0.5
                x0,x1,x2,y = input_x
                reward_payload = (x0, x1, x2, y, loss_mask)
                return reward_payload

            reinforce = reinforce_one

            s1_win = 0
            s2_win = 0
            def action_score_pr(pr_mark, before_logits, delete_logits, replace_logits, action):
                num_tag = np.count_nonzero(action)
                penalty = (num_tag - 4) * 0.1 if num_tag > 3 else 0

                s1 = replace_logits[0] - before_logits[0]
                s2 = replace_logits[0] - delete_logits[0]

                score = max(s1, s2)
                nonlocal s1_win, s2_win
                if s1 > s2 :
                    s1_win += 1
                elif s2 > s1:
                    s2_win += 1
                return score - penalty

            def select_best(alt_logits, instance_infos, deleted_mask_list):
                models_score_list = []
                reinforce_payload_list = []
                num_tag_list = []
                pos_win = 0
                pos_trial = 0
                for info in instance_infos:
                    init_logit = info['init_logits']
                    models_delete_logits = alt_logits[info['idx_delete_tagged']]
                    models_replace_logits = alt_logits[info['idx_delete_tagged']+1]
                    input_x = info['orig_input']
                    pr_mark = info['pr_mark'] # list[index]

                    predicted_action = deleted_mask_list[info['idx_delete_tagged']]
                    num_tag = np.count_nonzero(predicted_action)
                    num_tag_list.append(num_tag)
                    models_score = action_score_pr(pr_mark, init_logit, models_delete_logits, models_replace_logits, predicted_action)
                    models_score_list.append(models_score)
                    #self.log2.debug(
                    #    "target_ce_drop : {0:.4f}  n_token : {1}".format(target_ce_drop, num_tag))

                    good_action = predicted_action
                    best_score = models_score

                    alt_run_indice = info['indice_delete_random']
                    n_runs = len(alt_run_indice)
                    assert n_runs % 2 == 0
                    for i in range(0, n_runs, 2):
                        idx_delete_random = info['indice_delete_random'][i]
                        idx_replace_random = info['indice_delete_random'][i]
                        delete_logits = alt_logits[idx_delete_random]
                        replace_logits = alt_logits[idx_replace_random]
                        random_action = deleted_mask_list[idx_delete_random]
                        alt_score = action_score_pr(pr_mark, init_logit, delete_logits, replace_logits, random_action)
                        if alt_score > best_score :
                            best_score = alt_score
                            good_action = random_action

                    reward_payload = reinforce(good_action, input_x)
                    reinforce_payload_list.append(reward_payload)
                    if models_score >= best_score:
                        pos_win += 1
                    pos_trial += 1

                match_rate = pos_win / pos_trial
                avg_score = average(models_score_list)
                self.log.debug(" s1_win/s2_win = {}/{}".format(s1_win, s2_win))
                self.log.debug("[PR] drop score : {0:.4f}  suc_rate : {1:0.2f}".format(avg_score, match_rate))
                summary.value.add(tag='#Tags_PR', simple_value=average(num_tag_list))
                summary.value.add(tag='Score_PR', simple_value=avg_score)
                summary.value.add(tag='Success_PR', simple_value=match_rate)
                return reinforce_payload_list

            reinforce_payload = select_best(alt_logits, instance_infos, deleted_mask_list)


            def commit_reward(reinforce_payload):
                batches = get_batches_ex(reinforce_payload, self.hparam.batch_size, 5)
                rl_loss_list = []
                for batch in batches:
                    x0, x1, x2, y, loss_mask = batch
                    _, rl_loss, = self.sess.run([train_rl, task.rl_loss,
                                                                                ],
                                            feed_dict={
                                                task.x_list[0]: x0,
                                                task.x_list[1]: x1,
                                                task.x_list[2]: x2,
                                                task.y : y,
                                                task.rf_mask : loss_mask,
                                            })
                    rl_loss_list.append(rl_loss)
                return average(rl_loss_list)

            avg_rl_loss = commit_reward(reinforce_payload)

            summary.value.add(tag='PR_Loss', simple_value=avg_rl_loss)
            self.train_writer.add_summary(summary, self.g_step)
            self.train_writer.flush()

        def train_classification(batch, step_i):
            loss_val, summary, acc,  _ = self.sess.run([task.loss, self.merged, task.acc, train_cls,
                                             ],
                                            feed_dict=batch2feed_dict(batch)
                                            )
            self.log.debug("Step {0} train loss={1:.04f} acc={2:.03f}".format(step_i, loss_val, acc))
            self.train_writer.add_summary(summary, self.g_step)
            return loss_val, acc


        def train_fn(batch, step_i):
            loss_val, acc = train_classification(batch, step_i)
            train_explain(batch, step_i)
            self.g_step += 1
            return loss_val, acc


        def eval_acc():
            loss_list = []
            acc_list = []
            for batch in dev_batches[:10]:
                loss_val, summary, acc = self.sess.run([task.loss, self.merged, task.acc],
                                                  feed_dict=batch2feed_dict(batch)
                                                  )
                loss_list.append(loss_val)
                acc_list.append(acc)

                self.test_writer.add_summary(summary, self.g_step)
            self.log.info("Validation : loss={0:.04f} acc={1:.04f}".format(average(loss_list), average(acc_list)))

        def valid_fn():
            eval_acc()
            if exp_config.ex_val:
                eval_tag()

        def save_fn():
            self.save_model(exp_config.name, 1)


        valid_freq = 25
        print("Total of {} train batches".format(len(train_batches)))
        steps = int(len(train_batches) * 0.3)
        loss, _ = sub_step_runner(train_batches, train_fn,
                                  valid_fn, valid_freq,
                                  save_fn, self.save_interval,
                                  steps=steps)


    def nli_visualization(self, nli_setting, exp_config, data_loader, preload_id, explain_tag):
        print("nli_visualization")
        target_class = ENLIDef.get_target_class(explain_tag)
        enc_explain_dev, explain_dev = data_loader.get_dev_explain(explain_tag)
        task = transformer_nli(self.hparam, nli_setting.vocab_size, 1, False)
        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        if preload_id is not None:
            name = preload_id[0]
            id = preload_id[1]
            if exp_config.load_names :
                self.load_model_white(name, id, exp_config.load_names)
            else:
                assert False

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

            scores = eval_explain(conf_logit, data_loader, explain_tag)
            for metric in scores.keys():
                print("{}\t{}".format(metric, scores[metric]))
        eval()
        train_batches, train_batches_info, dev_batches = self.load_nli_data_with_info(data_loader)

        result = []
        #batches = get_batches_ex(enc_explain_dev, self.hparam.batch_size, 3)
        for batch in dev_batches[0:320]:
            x0, x1, x2, y = batch
            logits, conf_logit = self.sess.run([task.sout, task.conf_logits],
                                               feed_dict={
                                                   task.x_list[0]: x0,
                                                   task.x_list[1]: x1,
                                                   task.x_list[2]: x2,
                                               })
            predictions = logits.argmax(axis=1)


            for idx in range(len(x0)):
                input_ids = x0[idx]
                conf_p, conf_h = data_loader.split_p_h_with_input_ids(conf_logit[idx], input_ids)
                #prem, hypo, p_indice, h_indice = entry

                p_enc, h_enc = data_loader.split_p_h_with_input_ids(input_ids, input_ids)
                p_tokens = data_loader.encoder.decode_list(p_enc)
                h_tokens = data_loader.encoder.decode_list(h_enc)
                result.append((conf_p, conf_h, p_tokens, h_tokens, predictions[idx], y[idx]))

        #save_to_pickle(result, exp_config.name)
        visualize.visualize_nli(result, exp_config.name)
        #visualize.word_stat(result, exp_config.name)
        #visualize.make_explain_sentence(result, exp_config.name)



    def nli_visualization_pairing(self, nli_setting, exp_config, data_loader, preload_id):
        print("nli_visualization_pairing")
        task = transformer_nli(self.hparam, nli_setting.vocab_size, 6, False)
        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        if preload_id is not None:
            name = preload_id[0]
            id = preload_id[1]
            if exp_config.load_names :
                self.load_model_white(name, id, exp_config.load_names)
            else:
                assert False

        train_batches, train_batches_info, dev_batches = self.load_nli_data_with_info(data_loader)

        def sample_target_mask(batch):
            x0, x1, x2, y = batch

            def sample_it(input_ids, segment_ids):
                num_mark = 1
                length = len(segment_ids)
                last_valid = 0
                for i in range(length):
                    if segment_ids[i] > 0:
                        last_valid = i

                def sample_len():
                    l = 1
                    v = random.random()
                    while v < 0.25 and l < length:
                        l = l * 2
                    return min(l, length)

                indice = []
                for i in range(num_mark):
                    mark_len = sample_len()
                    start_idx = pick1(range(last_valid))
                    while input_ids[start_idx] == SEP_ID or input_ids[start_idx] == CLS_ID:
                        start_idx = pick1(range(last_valid))
                    end_idx = min(start_idx + mark_len, last_valid)

                    begin_segment_id = segment_ids[start_idx]
                    for idx in range(start_idx, end_idx):
                        if segment_ids[idx] != begin_segment_id:
                            break
                        if input_ids[idx] == SEP_ID:
                            break
                        indice.append(idx)

                for idx in indice:
                    segment_ids[idx] = ENLIDef.get_segment_marker(segment_ids[idx])

                return indice, segment_ids

            n_inst = len(batch)
            indice_list = []
            marked_inputs = []
            for i in range(n_inst):
                indice, x2_new = sample_it(x0[i], x2[i])
                marked_inputs.append((x0[i], x1[i], x2_new, y[i]))
                indice_list.append(indice)

            mark_batches = get_batches_ex(marked_inputs, self.hparam.batch_size, 4)
            return indice_list, mark_batches[0]

        result = []
        #batches = get_batches_ex(enc_explain_dev, self.hparam.batch_size, 3)
        for batch in dev_batches[20:100]:

            indice_list, mark_batch = sample_target_mask(batch)
            x0, x1, x2, y = mark_batch
            logits, conf_logit = self.sess.run([task.sout, task.conf_logits],
                                               feed_dict={
                                                   task.x_list[0]: x0,
                                                   task.x_list[1]: x1,
                                                   task.x_list[2]: x2,
                                               })
            predictions = logits.argmax(axis=1)

            print(self.hparam.batch_size)
            print(len(indice_list))
            print(len(x0))

            for idx in range(len(x0)):
                input_ids = x0[idx]
                conf_p, conf_h = data_loader.split_p_h_with_input_ids(conf_logit[idx], input_ids)
                #prem, hypo, p_indice, h_indice = entry

                p_enc, h_enc = data_loader.split_p_h_with_input_ids(input_ids, input_ids)
                p_tokens = data_loader.encoder.decode_list(p_enc)
                h_tokens = data_loader.encoder.decode_list(h_enc)

                if indice_list[idx][0] < len(p_tokens) + 1:
                    conf_p = np.zeros_like(conf_p)
                    for j in indice_list[idx]:
                        conf_p[j-1] = 1
                    conf_h = conf_h
                else: ## Mark was hypothesis
                    conf_p = conf_p
                    conf_h = np.zeros_like(conf_h)
                    offset = len(p_tokens) + 2
                    for j in indice_list[idx]:
                        conf_h[j-offset] = 1

                    result.append((conf_p, conf_h, p_tokens, h_tokens, predictions[idx], y[idx]))

        save_to_pickle(result, exp_config.name)
        visualize.visualize_nli(result, exp_config.name)
        #visualize.make_explain_sentence(result, exp_config.name)


    def nli_visualization_lm(self, nli_setting, exp_config, data_loader, preload_id, data):
        print("nli_visualization_lm")
        task = transformer_nli(self.hparam, nli_setting.vocab_size, 6, False)
        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        if preload_id is not None:
            name = preload_id[0]
            id = preload_id[1]
            if exp_config.load_names :
                self.load_model_white(name, id, exp_config.load_names)
            else:
                assert False

        result = []
        batches = get_batches_ex(data, self.hparam.batch_size, 3)
        for batch in batches:
            x0, x1, x2, = batch
            logits, conf_logit = self.sess.run([task.sout, task.conf_logits],
                                               feed_dict={
                                                   task.x_list[0]: x0,
                                                   task.x_list[1]: x1,
                                                   task.x_list[2]: x2,
                                               })
            predictions = logits.argmax(axis=1)

            print(self.hparam.batch_size)
            print(len(x0))

            for idx in range(len(x0)):
                input_ids = x0[idx]
                conf_p, conf_h = conf_logit[idx][1:100], conf_logit[idx][100:]
                #prem, hypo, p_indice, h_indice = entry

                p_enc, h_enc = input_ids[1:100], input_ids[100:]
                p_tokens = data_loader.encoder.decode_list(p_enc)
                h_tokens = data_loader.encoder.decode_list(h_enc)

                result.append((conf_p, conf_h, p_tokens, h_tokens, predictions[idx], 0))

        save_to_pickle(result, exp_config.name)
        visualize.visualize_nli(result, exp_config.name)
        visualize.make_explain_sentence(result, exp_config.name)



    def eval_fidelity(self, nli_setting, exp_config, data_loader, preload_id, explain_tag):
        method = 5
        task = transformer_nli(self.hparam, nli_setting.vocab_size, method, is_training=False)

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()

        self.load_model_white2(preload_id, exp_config.load_names)

        target_class = ENLIDef.get_target_class(explain_tag)

        def forward_runs(insts):
            alt_batches = get_batches_ex(insts, self.hparam.batch_size, 3)
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
            return alt_logits

        def fetch_contrib(insts):
            batches = get_batches_ex(insts, self.hparam.batch_size, 3)
            attribs = []
            for batch in batches:
                x0, x1, x2 = batch
                conf, = self.sess.run([task.conf_logits, ],
                                               feed_dict={
                                                task.x_list[0]: x0,
                                                task.x_list[1]: x1,
                                                task.x_list[2]: x2,
                                               })

                attribs.append(conf)
            attribs = np.concatenate(attribs)
            return attribs

        def batch2feed_dict(batch):
            x0, x1, x2, y = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y: y,
            }
            return feed_dict

        train_batches, train_batches_info, dev_batches = self.load_nli_data_with_info(data_loader)


        def flatten_filter_batches(batches, target_class):
            output = []
            for batch in batches:
                x0, x1, x2, y = batch
                for i in range(len(x0)):
                    if y[i] == target_class:
                        output.append((x0[i], x1[i], x2[i]))
            return output

        flat_dev_batches = flatten_filter_batches(dev_batches, target_class)[:2000]

        def valid_fn():
            loss_list = []
            acc_list = []
            for batch in dev_batches[:10]:
                loss_val, summary, acc = self.sess.run([task.loss, self.merged, task.acc],
                                                  feed_dict=batch2feed_dict(batch)
                                                  )
                loss_list.append(loss_val)
                acc_list.append(acc)

            self.log.info("Validation : loss={0:.04f} acc={1:.04f}".format(average(loss_list), average(acc_list)))

        valid_fn()
        #enc_explain_dev, explain_dev = data_loader.get_dev_explain(explain_tag)

        for contrib_method in ["sensitivity", "rl", "random"]:
            if contrib_method == "rl":
                contrib_score = fetch_contrib(flat_dev_batches)
            elif contrib_method == "sensitivity":
                contrib_score = np.stack(explain_by_deletion(flat_dev_batches, explain_tag, forward_runs))
            elif contrib_method == "random":
                contrib_score = np.stack(explain_by_random(flat_dev_batches, explain_tag, forward_runs))
            else:
                assert False

            def demo():
                l = len(contrib_score)
                for i in range(l):
                    x0, x1, x2 = flat_dev_batches[i]
                    tokens = data_loader.encoder.decode_list(x0)
                    for w in tokens:
                        print(w, end=" ")
                    print("")
                    for idx in np.flip(np.argsort(contrib_score[i]))[:20]:
                        print(idx, tokens[idx])
                    print("----")

            #demo()
            acc_list = eval_fidelity(contrib_score, flat_dev_batches, forward_runs, explain_tag)

            print(contrib_method)
            for num_delete in sorted(acc_list.keys()):
                print("{}\t{}".format(num_delete, acc_list[num_delete]))



    def eval_fidelity_gradient(self, nli_setting, exp_config, data_loader, preload_id, explain_tag):
        print("eval_fidelity_gradient")
        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())


        target_class = ENLIDef.get_target_class(explain_tag)

        train_batches, train_batches_info, dev_batches = self.load_nli_data_with_info(data_loader)
        def batch2feed_dict(batch):
            x0, x1, x2, y = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y: y,
            }
            return feed_dict


        def forward_runs(insts):
            alt_batches = get_batches_ex(insts, self.hparam.batch_size, 3)
            alt_logits = []
            for batch in alt_batches:
                x0, x1, x2 = batch
                enc, att = self.sess.run(emb_outputs, feed_dict=feed_end_input(batch))

                logits, = self.sess.run([task.sout, ],
                                               feed_dict={
                                                task.encoded_embedding_in: enc,
                                                task.attention_mask_in: att
                                               })

                alt_logits.append(logits)
            alt_logits = np.concatenate(alt_logits)
            return alt_logits

        def flatten_filter_batches(batches, target_class):
            output = []
            for batch in batches:
                x0, x1, x2, y = batch
                for i in range(len(x0)):
                    if y[i] == target_class:
                        output.append((x0[i], x1[i], x2[i]))
            return output

        flat_dev_batches = flatten_filter_batches(dev_batches, target_class)[:2000]

        from attribution.gradient import explain_by_gradient
        self.sess = self.init_sess()
        from attribution.deepexplain.tensorflow import DeepExplain
        with DeepExplain(session=self.sess, graph=self.sess.graph) as de:
            task = transformer_nli_embedding_in(self.hparam, nli_setting.vocab_size, False)
            self.sess.run(tf.global_variables_initializer())
            self.load_model_white2(preload_id, exp_config.load_names)

            emb_outputs = task.encoded_embedding_out, task.attention_mask_out
            emb_input = task.encoded_embedding_in, task.attention_mask_in
            softmax_out = task.sout
            def feed_end_input(batch):
                x0, x1, x2 = batch
                return {task.x_list[0]:x0,
                        task.x_list[1]:x1,
                        task.x_list[2]:x2,
                        }


            #enc_explain_dev, explain_dev = data_loader.get_dev_explain(explain_tag)

            contrib_method = "intgrad" #"grad*input"#"saliency"#

            contrib_score = explain_by_gradient(flat_dev_batches, contrib_method, explain_tag, self.sess, de,
                                           feed_end_input, emb_outputs, emb_input, softmax_out)

            acc_list = eval_fidelity(contrib_score, flat_dev_batches, forward_runs, explain_tag)

            print(contrib_method)
            for num_delete in sorted(acc_list.keys()):
                print("{}\t{}".format(num_delete, acc_list[num_delete]))

    def gen_ubuntu_data(self, data_loader):

        all_len = 12724
        intervals = []
        st = 0
        while st < all_len:
            ed = st + 500
            intervals.append((st, ed))
            st = ed

        all_data = []
        with ProcessPoolExecutor(max_workers=32) as executor:
            future_list = []
            for interval in intervals:
                future = executor.submit(ubuntu.gen_ubuntu_data_part, interval)
                future_list.append((interval, future))

            for interval,future in future_list:
                train_insts = future.result()
                all_data += train_insts
        save_to_pickle(all_data, "ubuntu_train")

    def test_valid_ubuntu(self, data_loader):
        dev_data = data_loader.get_dev_data()
        dev_runs, golds = data_loader.flatten_payload(dev_data)
        preds = []

        for gold in golds:
            inst_size = len(gold)
            scores = [random.random() for _ in range(inst_size)]
            print(gold)
            pred = np.flip(np.argsort(scores))
            print(pred)
            preds.append(pred)

        assert len(preds) == len(golds)
        map_score = MAP_rank(preds, golds)
        print("random MAP : ", map_score)

    def train_ubuntu(self, exp_config, data_loader, preload_id):
        tprint("train_ubuntu")
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
        self.load_model_white2(preload_id, exp_config.load_names)
        tprint("Loading data...")
        dev_data = data_loader.get_dev_data()
        dev_runs, golds = data_loader.flatten_payload(dev_data)
        dev_batches = get_batches_ex(dev_runs, self.hparam.batch_size, 3)

        #train_data = data_loader.get_train_data()
        #train_batches = get_batches_ex(train_data, self.hparam.batch_size, 3)

        def valid_fn():
            logits_all = []
            for idx, batch in enumerate(dev_batches):
                x0, x1, x2 = batch
                feed_dict = {
                    task.x_list[0]: x0,
                    task.x_list[1]: x1,
                    task.x_list[2]: x2,
                }

                logits, = self.sess.run([task.logits],
                                                  feed_dict=feed_dict
                                                  )

                logits = np.reshape(logits, [-1])
                logits_all += list(logits)

            idx = 0
            preds = []
            for gold in golds:
                inst_size = len(gold)
                scores = logits_all[idx:idx+inst_size]
                pred = np.argsort(scores)
                print(pred)
                print(gold)
                preds.append(pred)
                idx = idx + inst_size

                assert len(pred) == inst_size
            map_score = MAP_rank(preds, golds)

            self.log.info("Validation : map={0:.02f}".format(map_score))
            summary = tf.Summary()
            summary.value.add(tag='MAP', simple_value=map_score)
            self.test_writer.add_summary(summary, self.g_step)
            self.test_writer.flush()


        def logits2acc(logits):
            paired = np.reshape(logits, [-1, 2])
            n = paired.shape[0]
            acc = np.count_nonzero((paired[: ,1] - paired[:, 0]) > 0)
            return acc / n

        def train_fn(batch, step_i):
            # normal train
            loss_val, summary, logits, _= self.sess.run([task.loss, self.merged, task.logits, train_op],
                                               feed_dict=batch2feed_dict(batch)
                                               )
            acc = logits2acc(logits)
            self.log.debug("Step {0} train loss={1:.04f} acc={2:.02f}".format(step_i, loss_val, acc))
            self.train_writer.add_summary(summary, self.g_step)
            summary = tf.Summary()
            summary.value.add(tag='acc', simple_value=acc)
            self.train_writer.add_summary(summary, self.g_step)
            self.train_writer.flush()
            self.g_step += 1
            return loss_val, 0

        def batch2feed_dict(batch):
            x0, x1, x2 = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
            }
            return feed_dict

        def save_fn():
            self.save_model(exp_config.name, 1)

        print("Start Training")
        valid_freq = 333
        valid_fn()
        for i in range(exp_config.num_epoch):
            for data_id in range(0,26):
                train_batches = load_from_pickle("ubuntu_train_batch_16_{}".format(data_id))
                loss, _ = epoch_runner(train_batches, train_fn,
                                       valid_fn, valid_freq,
                                       save_fn, self.save_interval)
                self.log.info("[Train] Epoch {}-{} Done. Loss={}".format(i, data_id, loss))

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
            fout = cpath.open_pred_output("rerank_{}".format(exp_config.name))
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


    def train_adhoc(self, exp_config, data_loader, preload_id):
        tprint("train_adhoc")
        task = transformer_adhoc2(self.hparam, data_loader.voca_size)
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
        dev_batches = get_batches_ex(data_loader.get_dev_data(), self.hparam.batch_size, 3)

        encoder_unit = data_loader.encoder_unit

        def enc_query_doc_to_payload(query_docs_list):
            with ProcessPoolExecutor(max_workers=8) as executor:
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
                    print("... Done")
            return enc_payload

        def get_score_from_enc_payload(enc_payload, query_docs_list):
            def eval(runs):
                data = []
                for entry in runs:
                    data.append((entry['input_ids'], entry['input_mask'], entry['segment_ids'], 0))
                batches = get_batches_ex(data, batch_size, 4)
                y_list = []
                for batch in batches:
                    y, = self.sess.run([task.y, ],
                                       feed_dict=batch2feed_dict(batch)
                                       )
                    y_list.append(y)
                ys = np.concatenate(y_list)
                return ys


            pk = PromiseKeeper(eval)
            score_list_future = []
            for query, doc_id, runs in enc_payload:
                y_futures = list([MyPromise(x, pk).future() for x in runs])
                score_list_future.append((query, doc_id, y_futures))

            pk.do_duty()

            per_query = defaultdict(list)
            for query, doc_id, y_futures in score_list_future:
                per_query[query].append((doc_id, max_future(y_futures)))

            result = []
            for query, _ in query_docs_list:
                result.append(per_query[query])
            return result

        collection, idf, inv_index, tf_index, _ = load_trec_data_proc()
        queries = list(load_marco_queries())

        def bm25_run():
            # compare score
            target_queries = queries[300:350]
            result = []
            for q in target_queries:
                score = Counter()
                print(q)
                for q_term in q.split():
                    if q_term in tf_index:
                        for doc_id, tf in tf_index[q_term].items():
                            score[doc_id] += tf * idf[q_term]

                top_docs = list(score.most_common(100))
                result.append((q, top_docs))
            return result

        tprint("runing bm25...")
        bm25_result_w_query = bm25_run()

        def get_test_candidate_by_bm25():
            query_docs_list = []
            for q, top_docs in bm25_result_w_query:
                docs = list([(doc_id, collection[doc_id]) for doc_id in left(top_docs)])
                query_docs_list.append((q, docs))
            return query_docs_list

        query_docs_list = get_test_candidate_by_bm25()
        enc_query_doc = None
        def compare_bm25():
            tprint("compare bm25")
            nonlocal enc_query_doc
            if enc_query_doc is None:
                enc_query_doc = enc_query_doc_to_payload(query_docs_list)

            bm25_result = right(bm25_result_w_query)
            score_result = get_score_from_enc_payload(enc_query_doc, query_docs_list)
            common_sizes = defaultdict(list)
            for query_idx, q_result in enumerate(score_result):
                assert len(q_result) == len(bm25_result[query_idx])
                q_result = list(q_result)
                q_result.sort(key=lambda x:x[1], reverse=True)

                for k in [10,30,50]:
                    topk_nn = set(left(q_result[:k]))
                    topk_bm25 = set(left(bm25_result[query_idx][:k]))
                    common_in_k = len(topk_nn.intersection(topk_bm25))
                    common_sizes[k].append(common_in_k)
            for k in common_sizes:
                print("Common at {} : {}".format(k, average(common_sizes[k])))



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
                task.y: y,
            }
            return feed_dict

        def save_fn():
            self.save_model(exp_config.name, 100)



        def generate_train_batch():
            working_unit = 2
            data_size = batch_size * working_unit
            batches = get_batches_ex(data_loader.get_train_data(data_size), batch_size, 4)
            return batches

        queue_feader = QueueFeader(300, generate_train_batch, True)

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

            batch = queue_feader.get()
            train_fn(batch, step_i)
            self.g_step += 1



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
            acc_list = []
            loss_list = []
            for idx, batch in enumerate(dev_batches):
                loss_val, summary, logits = self.sess.run([task.loss, self.merged, task.logits],
                                                  feed_dict=batch2feed_dict(batch)
                                                  )
                acc_list.append(logits2acc(logits))
                loss_list.append(loss_val)
                v_step = self.g_step + idx - int(len(dev_batches) / 2)
                self.test_writer.add_summary(summary, v_step)

            acc = average(acc_list)
            self.log.info("Validation : loss={0:.04f} acc={1:.02f}".
                          format(average(loss_list), acc))
            summary = tf.Summary()
            summary.value.add(tag='acc', simple_value=acc)
            self.test_writer.add_summary(summary, self.g_step)
            self.test_writer.flush()


        def logits2acc(logits):
            paired = np.reshape(logits, [-1, 2])
            n = paired.shape[0]
            acc = np.count_nonzero((paired[: ,1] - paired[:, 0]) > 0)
            return acc / n

        def train_fn(batch, step_i):
            # normal train
            loss_val, summary, logits, _= self.sess.run([task.loss, self.merged, task.logits, train_op],
                                               feed_dict=batch2feed_dict(batch)
                                               )
            acc = logits2acc(logits)
            self.log.debug("Step {0} train loss={1:.04f} acc={2:.02f}".format(step_i, loss_val, acc))
            self.train_writer.add_summary(summary, self.g_step)
            summary = tf.Summary()
            summary.value.add(tag='acc', simple_value=acc)
            self.train_writer.add_summary(summary, self.g_step)
            self.train_writer.flush()
            self.g_step += 1
            return loss_val, 0

        def batch2feed_dict(batch):
            x0, x1, x2 = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
            }
            return feed_dict

        def save_fn():
            self.save_model(exp_config.name, 1)

        print("dev")
        valid_freq = 25
        last_save = time.time()
        max_step = 1000 * 1000 * 1000
        for step_i in range(max_step):
            if step_i % valid_freq == 0:
                valid_fn()

            if save_fn is not None:
                if time.time() - last_save > exp_config.save_interval:
                    save_fn()
                    if exp_config.save_interval < 120 * 60:
                        exp_config.save_interval += 5 * 60
                    last_save = time.time()

            batch = data_loader.get_train_batch()
            train_fn(batch, step_i)
            self.g_step += 1


    def train_adhoc_ex(self, exp_config, data_loader, preload_id):
        tprint("train_adhoc_ex")
        #task = transformer_adhoc(self.hparam, data_loader.voca_size)
        task = transformer_adhoc_ex(self.hparam, data_loader.voca_size)
        with tf.variable_scope("optimizer"):
            train_op = self.get_train_op(task.loss * 4)
            train_rl = self.get_train_op(task.rl_loss, name="rl")

            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.9, beta2=0.98, epsilon=1e-8,
                                               name="bias")
            train_bias = optimizer.minimize(task.hinge_loss , global_step=self.global_step, var_list=[task.bias])

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
            acc_list = []
            loss_list = []
            for idx, batch in enumerate(dev_batches):
                loss_val, summary, logits = self.sess.run([task.loss, self.merged, task.logits],
                                                  feed_dict=batch2feed_dict(batch)
                                                  )
                acc_list.append(logits2acc(logits))
                loss_list.append(loss_val)
                v_step = self.g_step + idx - int(len(dev_batches) / 2)
                self.test_writer.add_summary(summary, v_step)


            self.log.info("Validation : loss={0:.04f} acc={1:.02f}".
                          format(average(loss_list), average(acc_list)))

        def logits2acc(logits):
            paired = np.reshape(logits, [-1, 2])
            n = paired.shape[0]
            acc = np.count_nonzero((paired[: ,1] - paired[:, 0]) > 0)
            return acc / n

        def train_explain(batch, step_i):
            summary = tf.Summary()

            def random_token():
                return random.randrange(10, data_loader.voca_size-1)

            def over_zero(np_arr):
                return np.less(0, np_arr).astype(np.float32)

            def sample_size():
                #prob = [(1, 0.5), (2, 0.2), (3, 0.1), (4, 0.1), (5, 0.1)]
                prob = [(1,0.9), (2,0.1)]
                v = random.random()
                for n, p in prob:
                    v -= p
                    if v < 0:
                        return n
                return 1
            logit2tag = over_zero
            def make_query_mask(batch):
                input_ids, input_mask, segment_ids = batch
                q_base = np.equal(segment_ids, 0).astype(int) * input_mask


                for i in range(len(q_base)):
                    for j in range(self.hparam.seq_max):
                        if input_ids[i,j] == SEP_ID:
                            q_base[i,j] = 0
                    q_base[i, 0] = 0
                return input_ids, input_mask, segment_ids, q_base.astype(float)

            def forward_runs(insts):
                alt_batches = get_batches_ex(insts, self.hparam.batch_size, 3)
                alt_logits = []
                for batch in alt_batches:
                    x0, x1, x2 = batch
                    logits, = self.sess.run([task.logits, ],
                                            feed_dict={
                                                task.x_list[0]: x0,
                                                task.x_list[1]: x1,
                                                task.x_list[2]: x2,
                                            })

                    alt_logits.append(logits)
                alt_logits = np.concatenate(alt_logits)
                return alt_logits

            ## Step 1) Prepare deletion RUNS
            def generate_alt_runs(batch):
                x0, x1, x2, q_mask = make_query_mask(batch)
                logits, ex_logit = self.sess.run([task.logits, task.ex_logits],
                                                 feed_dict={
                                                     task.x_list[0]: x0,
                                                     task.x_list[1]: x1,
                                                     task.x_list[2]: x2,
                                                     task.q_mask : q_mask,
                                                 })


                compare_deletion_num = 20
                instance_infos = []
                new_batches = []
                deleted_mask_list = []
                tag_size_list = []
                for i in range(len(logits)):
                    if True:
                        info = {}
                        info['init_logit'] = logits[i]
                        info['orig_input'] = (x0[i], x1[i], x2[i])
                        ex_tags = logit2tag(ex_logit[i])
                        self.log2.debug("EX_Score : {}".format(numpy_print(ex_logit[i])))
                        tag_size = np.count_nonzero(ex_tags)
                        tag_size_list.append(tag_size)
                        if tag_size > 10:
                            self.log2.debug("#Tagged token={}".format(tag_size))

                        info['idx_delete_tagged'] = len(new_batches)
                        new_batches.append(token_replace(ex_tags, x0[i], x1[i], x2[i], random_token))
                        deleted_mask_list.append(ex_tags)

                        indice_delete_random = []

                        for _ in range(compare_deletion_num):
                            tag_size = sample_size()
                            indice_delete_random.append(len(new_batches))
                            x_list, delete_mask = random_delete_with_mask(tag_size, x0[i], x1[i], x2[i], q_mask[i])
                            new_batches.append(x_list)
                            deleted_mask_list.append(delete_mask)

                        info['indice_delete_random'] = indice_delete_random
                        instance_infos.append(info)
                if tag_size_list:
                    avg_tag_size = average(tag_size_list)
                    self.log2.debug("avg Tagged token#={}".format(avg_tag_size))
                return new_batches, instance_infos, deleted_mask_list

            new_batches, instance_infos, deleted_mask_list = generate_alt_runs(batch)

            if not new_batches:
                return
            ## Step 2) Execute deletion Runs
            alt_logits = forward_runs(new_batches)

            def reinforce_one(good_action, input_x):
                pos_reward_indice = np.int_(good_action)
                self.log2.debug("Reward Tokens : {}".format(numpy_print(pos_reward_indice[:20])))
                loss_mask = +pos_reward_indice + np.ones_like(pos_reward_indice) * (-0.1)
                x0, x1, x2 = input_x
                reward_payload = (x0, x1, x2, loss_mask)
                return reward_payload

            def action_score(before_prob, after_prob, action):
                num_tag = np.count_nonzero(action)
                penalty = (num_tag - 1) * 0.2 if num_tag > 1 else 0
                score = before_prob - after_prob
                score = score - penalty
                return score

            ## Step 3) Calc reward
            def calc_reward(alt_logits, instance_infos, deleted_mask_list):
                models_movement_list = []
                reinforce_payload_list = []
                pos_win = 0
                pos_trial = 0
                for info in instance_infos:
                    init_output = info['init_logit']
                    models_after_output = alt_logits[info['idx_delete_tagged']]
                    input_x = info['orig_input']

                    predicted_action = deleted_mask_list[info['idx_delete_tagged']]
                    models_movement = action_score(init_output, models_after_output, predicted_action)
                    models_movement_list.append(models_movement)


                    good_action = predicted_action
                    best_movement = models_movement
                    for idx_delete_random in info['indice_delete_random']:
                        alt_after_output = alt_logits[idx_delete_random]
                        random_action = deleted_mask_list[idx_delete_random]
                        alt_movement = action_score(init_output, alt_after_output, random_action)
                        if alt_movement > best_movement:
                            best_movement = alt_movement
                            good_action = random_action

                    self.log2.debug("Reward : model={0:.2f} best={1:.2f} ".format(models_movement, best_movement))
                    reward_payload = reinforce_one(good_action, input_x)
                    reinforce_payload_list.append(reward_payload)
                    if models_movement >= best_movement:
                        pos_win += 1
                    pos_trial += 1

                match_rate = pos_win / pos_trial
                avg_score = average(models_movement_list)
                self.log.debug("drop score : {0:.4f}  suc_rate : {1:0.2f}".format(avg_score, match_rate))
                summary.value.add(tag='CE_Drop', simple_value=avg_score)
                summary.value.add(tag='Success', simple_value=match_rate)
                return reinforce_payload_list

            reinforce_payload = calc_reward(alt_logits, instance_infos, deleted_mask_list)

            def commit_reward(reinforce_payload):
                batches = get_batches_ex(reinforce_payload, self.hparam.batch_size, 4)
                rl_loss_list = []
                for batch in batches:
                    x0, x1, x2, rf_mask = batch
                    x0, x1, x2, q_mask = make_query_mask((x0,x1,x2))
                    _, _, rl_loss, h_loss, bias, ex_logits, = self.sess.run([train_rl, train_bias,
                                                                       task.rl_loss, task.hinge_loss,
                                                                       task.bias,
                                                              task.ex_logits,
                                                              ],
                                                             feed_dict={
                                                                 task.x_list[0]: x0,
                                                                 task.x_list[1]: x1,
                                                                 task.x_list[2]: x2,
                                                                 task.rf_mask: rf_mask,
                                                                 task.q_mask: q_mask
                                                             })
                    self.log2.debug("rl_loss= {0:.2f} hinge_loss={1:.2f} bias={2:.2f} ".format(rl_loss, h_loss, bias))
                    rl_loss_list.append(rl_loss)
                return average(rl_loss_list)

            ## Step 4) Update gradient
            avg_rl_loss = commit_reward(reinforce_payload)

            summary.value.add(tag='RL_Loss', simple_value=avg_rl_loss)
            self.train_writer.add_summary(summary, self.g_step)
            self.train_writer.flush()

        def train_score(batch, step_i):
            # normal train
            loss_val, summary, logits, _= self.sess.run([task.loss, self.merged, task.logits, train_op],
                                               feed_dict=batch2feed_dict(batch)
                                               )
            acc = logits2acc(logits)
            self.log.debug("Step {0} train loss={1:.04f} acc={2:.02f}".format(step_i, loss_val, acc))
            self.train_writer.add_summary(summary, self.g_step)
            summary = tf.Summary()
            summary.value.add(tag='acc', simple_value=acc)
            self.train_writer.add_summary(summary, self.g_step)
            self.train_writer.flush()
            self.g_step += 1
            return loss_val, 0

        def train_fn(batch, step_i):
            train_explain(batch, step_i)
            return train_score(batch, step_i)

        def batch2feed_dict(batch):
            x0, x1, x2 = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
            }
            return feed_dict

        def save_fn():
            self.save_model(exp_config.name, 30)

        print("dev")
        valid_freq = 25
        last_save = time.time()
        max_step = 1000 * 1000 * 1000
        for step_i in range(max_step):
            if step_i % valid_freq == 0:
                valid_fn()

            if save_fn is not None:
                if time.time() - last_save > exp_config.save_interval:
                    save_fn()
                    if exp_config.save_interval < 120 * 60:
                        exp_config.save_interval += 5 * 60
                    last_save = time.time()

            batch = data_loader.get_train_batch()
            train_fn(batch, step_i)
            self.g_step += 1



    def train_score_merger(self, exp_config, data_loader, preload_id = None):
        tprint("train_score_merger")
        #task = ScoreCombinerMax(self.hparam)
        task = ScoreCombinerFF(self.hparam)
        with tf.variable_scope("optimizer"):
            train_op = self.get_train_op(task.loss)
        self.log.name = exp_config.name
        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)
        batch_size = self.hparam.batch_size
        if preload_id is not None:
            name = preload_id[0]
            id = preload_id[1]
            self.load_model_all(name, id)

        tprint("get_dev_data...")
        dev_batches = data_loader.get_dev_data(batch_size)

        def add_tf_summary(writer, tag, value):
            summary = tf.Summary()
            summary.value.add(tag=tag, simple_value=value)
            writer.add_summary(summary, self.g_step)
            writer.flush()

        def valid_fn():
            #compare_bm25()
            acc_list = []
            loss_list = []
            for idx, batch in enumerate(dev_batches):
                loss_val, summary, logits = self.sess.run([task.loss, self.merged, task.logits],
                                                  feed_dict=batch2feed_dict(batch)
                                                  )
                acc_list.append(logits2acc(logits))
                loss_list.append(loss_val)
                v_step = self.g_step + idx - int(len(dev_batches) / 2)
                self.test_writer.add_summary(summary, v_step)

            acc = average(acc_list)
            self.log.info("Validation : loss={0:.04f} acc={1:.02f}".
                          format(average(loss_list), acc))
            add_tf_summary(self.test_writer, 'acc', acc)


        def logits2acc(logits):
            paired = np.reshape(logits, [-1, 2])
            n = paired.shape[0]
            acc = np.count_nonzero((paired[: ,1] - paired[:, 0]) > 0)
            return acc / n

        def train_fn(batch, step_i):
            # normal train
            loss_val, summary, logits, _= self.sess.run([task.loss, self.merged, task.logits, train_op],
                                               feed_dict=batch2feed_dict(batch)
                                               )
            acc = logits2acc(logits)
            self.log.debug("Step {0} train loss={1:.04f} acc={2:.02f}".format(step_i, loss_val, acc))
            self.train_writer.add_summary(summary, self.g_step)
            add_tf_summary(self.train_writer, 'acc', acc)

            self.g_step += 1
            return loss_val, 0

        def batch2feed_dict(batch):
            x0, x1, x2 = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
            }
            return feed_dict

        def save_fn():
            self.save_model(exp_config.name, 1)


        print("dev")
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

            batch = data_loader.get_train_batch(batch_size)

            train_fn(batch, step_i)
            self.g_step += 1


    def predict_robust(self, exp_config, voca_size, preload_id, payload_path, task_idx):
        tprint("predict_robust")
        if exp_config.name.startswith("Adhoc_J"):
            task = transformer_adhoc(self.hparam, voca_size)
        elif exp_config.name.startswith("Adhoc_K"):
            task = transformer_adhoc2(self.hparam, voca_size)
        else:
            assert False

        self.log.name = exp_config.name
        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        batch_size = self.hparam.batch_size
        def batch2feed_dict(batch):
            x0, x1, x2 = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
            }
            return feed_dict

        tprint("Loading Model...")
        if preload_id is not None:
            name = preload_id[0]
            id = preload_id[1]
            if exp_config.load_names:
                self.load_model_white(name, id, exp_config.load_names)
            else:
                self.load_model_bert(name, id)

        payload = pickle.load(open(payload_path, "rb"))

        def eval(runs):
            data = []
            for entry in runs:
                data.append((entry['input_ids'], entry['input_mask'], entry['segment_ids']))
            tprint("Packing batches (batch_size={})".format(batch_size))
            batches = get_batches_ex(data, batch_size, 3)
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

        q_id_list = [
            (301, 325),
            (326, 350),
            (351, 375),
            (376, 400),
            (401, 425),
            (426, 450),
            (601, 625),
            (626, 650),
            (651, 675),
            (676, 700),
        ]

        st, ed = q_id_list[task_idx]

        def q_range(q_id):
            return st <= int(q_id) <= ed

        pk = PromiseKeeper(eval)
        score_list_future = []
        for doc_id, q_id, runs in payload:
            if q_range(q_id):
                y_futures = list([MyPromise(x, pk).future() for x in runs])
                score_list_future.append((q_id, doc_id, y_futures))

        pk.do_duty()
        tprint("Completed GPU computations")
        per_query = defaultdict(list)
        f_out_log = cpath.open_pred_output("detail_rerank_{}_{}_{}".format(exp_config.name, st, ed))
        for q_id, doc_id, y_futures in score_list_future:
            scores = list([f.get() for f in y_futures])
            f_out_log.write("{} {} ".format(q_id, doc_id) + " ".join([str(s) for s in scores]) + "\n")
            per_query[q_id].append((doc_id, max(scores)))

        fout = cpath.open_pred_output("rerank_{}_{}_{}".format(exp_config.name, st, ed))
        for q_id in per_query:
            q_result = per_query[q_id]
            q_result.sort(key=lambda x: x[1], reverse=True)

            rank_idx = 1
            for doc_id, score in q_result:
                fout.write("{} Q0 {} {} {} galago\n".format(q_id, doc_id, rank_idx, score[0]))
                rank_idx += 1




    def predict_robust_L_part1(self, exp_config, voca_size, preload_id, payload_path, interval):
        tprint("predict_robust_with_max_pool")
        task = transformer_adhoc_ex(self.hparam, voca_size)
        self.log.name = exp_config.name
        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        batch_size = self.hparam.batch_size
        def batch2feed_dict(batch):
            x0, x1, x2 = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
            }
            return feed_dict

        tprint("Loading Model...")
        if preload_id is not None:
            name = preload_id[0]
            id = preload_id[1]
            if exp_config.load_names:
                self.load_model_white(name, id, exp_config.load_names)
            else:
                self.load_model_bert(name, id)

        payload = pickle.load(open(payload_path, "rb"))

        def eval(runs):
            data = []

            for entry in runs:
                data.append((entry['input_ids'], entry['input_mask'], entry['segment_ids']))
            tprint("Packing batches (batch_size={})".format(batch_size))
            batches = get_batches_ex(data, batch_size, 3)
            tprint("Runing neural network prediction (#batch={})".format(len(batches)))
            y_list = []
            enc_list = []
            def np_append(dst_lst, np_arr):
                for i in range(len(np_arr)):
                    dst_lst.append(np_arr[i])


            def fetch_query_part(cl , batch):
                input_ids, input_mask, segment_ids = batch
                q_max = 20
                valid_part = []
                for i in range(len(cl)):
                    for j in range(q_max):
                        if input_ids[i, j] == SEP_ID:
                            break

                    valid_part.append(cl[i, 1:j])
                return valid_part

            ticker = TimeEstimator(len(batches), sample_size=20)
            for batch in batches:
                y, cl= self.sess.run([task.logits, task.cl],
                                   feed_dict=batch2feed_dict(batch)
                                   )
                np_append(y_list, y)

                cl = fetch_query_part(cl, batch)
                np_append(enc_list, cl)
                ticker.tick()
            return list(zip(y_list, enc_list))


        st, ed = interval
        def q_range(q_id):
            return st <= int(q_id) <= ed

        pk = PromiseKeeper(eval)
        score_list_future = []
        for doc_id, q_id, runs in payload:
            if q_range(q_id):
                y_futures = list([MyPromise(x, pk).future() for x in runs])
                score_list_future.append((q_id, doc_id, y_futures))

        pk.do_duty()
        tprint("Completed GPU computations")
        self.sess.close()
        tf.reset_default_graph()

        tail = "middle_rerank_{}_{}_{}".format(exp_config.name, st, ed)

        fout = open(os.path.join(cpath.prediction_dir, tail), "wb")

        middle = []
        for q_id, doc_id, y_futures in score_list_future:
            def fetch(f):
                y, v = f.get()
                return y

            p = list([f.get() for f in y_futures])

            entry = (q_id, doc_id, p)
            middle.append(entry)

        pickle.dump(middle, fout)
        return score_list_future

    def predict_robust_L_part2(self, exp_config, score_list_future, preload_id2, interval):
        ## -------------------------------------------------- ##
        hp2 = HPMerger()
        task = ScoreCombinerFF(hp2)
        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()

        def batch2feed_dict(batch):
            x0, x1, x2 = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
            }
            return feed_dict

        if preload_id2 is not None:
            name = preload_id2[0]
            id = preload_id2[1]
            run_dir = os.path.join(self.model_dir, 'runs')
            save_dir = os.path.join(run_dir, name)
            fullpath = os.path.join(save_dir, "{}".format(id))

            variables = tf.contrib.slim.get_variables_to_restore()
            print("All Variable")
            for v in variables:
                print(v)

            def condition(v):
                return ("dense" in v.name) or ('LayerNorm' in v.name)
            variables_to_restore = [v for v in variables if condition(v)]
            for v in variables_to_restore:
                print(v)

            print("Restoring: {} {}".format(name, id))
            self.loader = tf.train.Saver(variables_to_restore, max_to_keep=1)
            self.loader.restore(self.sess, fullpath)


        def eval_pool(entry_list):
            batches = get_batches_ex(entry_list, hp2.batch_size, 3)
            y_list = []
            for batch in batches:
                y, = self.sess.run([task.logits,],
                                         feed_dict=batch2feed_dict(batch)
                                   )
                y_list.append(y)
            y_list = np.concatenate(y_list)
            return y_list

        pk2 = PromiseKeeper(eval_pool)
        per_query = defaultdict(list)
        for q_id, doc_id, y_futures in score_list_future:
            def to_vector(f):
                y, v = f.get()
                return [y] + list(v)

            vectors = np.stack(list([to_vector(f) for f in y_futures])) # [num_span, hidden_size]

            entry = score_loader.encode(vectors, hp2.seq_max, hp2.hidden_units)
            per_query[q_id].append((doc_id, MyPromise(entry, pk2).future()))
        pk2.do_duty()
        st, ed = interval
        fout = cpath.open_pred_output("rerank_{}_{}_{}".format(exp_config.name, st, ed))
        for q_id in per_query:
            q_result_f = per_query[q_id]
            q_result = list([(doc_id, future.get()) for doc_id, future in q_result_f])
            q_result.sort(key=lambda x: x[1], reverse=True)

            rank_idx = 1
            for doc_id, score in q_result:
                fout.write("{} Q0 {} {} {} galago\n".format(q_id, doc_id, rank_idx, score[0]))
                rank_idx += 1
        print("Done")

    def predict_for_pooling(self, exp_config, voca_size, preload_id, payload_path):
        tprint("predict_for_pooling")
        task = transformer_adhoc_ex(self.hparam, voca_size)
        self.log.name = exp_config.name
        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        batch_size = self.hparam.batch_size

        def batch2feed_dict(batch):
            x0, x1, x2 = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
            }
            return feed_dict
        payload = pickle.load(open(payload_path, "rb"))

        tprint("Loading Model...")
        if preload_id is not None:
            name = preload_id[0]
            id = preload_id[1]
            if exp_config.load_names:
                self.load_model_white(name, id, exp_config.load_names)
            else:
                self.load_model_bert(name, id)


        def eval(inputs):
            batches = get_batches_ex(inputs, batch_size, 3)
            ticker = TimeEstimator(len(batches), sample_size=20)
            y_list = []
            enc_list = []

            def np_append(dst_lst, np_arr):
                for i in range(len(np_arr)):
                    dst_lst.append(np_arr[i])

            def fetch_query_part(cl , batch):
                input_ids, input_mask, segment_ids = batch
                q_max = 20
                valid_part = []
                for i in range(len(cl)):
                    for j in range(q_max):
                        if input_ids[i, j] == SEP_ID:
                            break

                    valid_part.append(cl[i, 1:j])
                return valid_part

            for batch in batches:
                y, cl = self.sess.run([task.logits, task.cl],
                                   feed_dict=batch2feed_dict(batch)
                                   )
                np_append(y_list, y)

                cl = fetch_query_part(cl, batch)
                np_append(enc_list, cl)
                ticker.tick()
            return list(zip(y_list, enc_list))

        pk = PromiseKeeper(eval)
        f_result = []
        for run1, run2 in payload:
            futures_1 = []
            for e in run1:
                futures_1.append(MyPromise(e, pk).future())
            futures_2 = []
            for e in run2:
                futures_2.append(MyPromise(e, pk).future())
            f_result.append((futures_1, futures_2))
        pk.do_duty()

        result = []
        for r1, r2 in f_result:
            result.append((list_future(r1), list_future(r2)))

        save_name = payload_path + ".output"
        pickle.dump(result, open(save_name, "wb"))


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


    def rank_robust_bm25(self):
        queries = load_robust04_title_query()
        ranked_lists = load_2k_rank()

        collection_len = 252359881

        if True:
            collection = trec.load_robust(trec.robust_path)
            #inv_index = get_inverted_index(collection)
            #sample_key = random.sample(list(collection.keys()), 40000)
            #sample_collection = [collection[k] for k in sample_key]
            #idf = Idf(list(sample_collection))
            #ctf = collection_tf(list(sample_collection))
            #save_to_pickle(ctf, "ctf")
            #save_to_pickle(idf, "idf")
            idf = load_from_pickle("idf")
            #tf_index = get_tf_index(inv_index, True)
            #save_to_pickle(tf_index, "tf_index")
        else:
            mem_path = "/dev/shm/robust04.pickle"
            data_sampler = pickle.load(open(mem_path, "rb"))
            collection = data_sampler.collection
            inv_index = data_sampler.inv_index
            tf_index = get_tf_index(inv_index, True)
            idf = data_sampler.idf
        top_k = 100
        avdl = collection_len / len(collection)
        print("len(collection)", len(collection))
        print("avdl ", avdl)
        fout = cpath.open_pred_output("rerank_{}".format("bm25"))
        for q_id in ranked_lists:
            ranked = ranked_lists[q_id]
            ranked.sort(key=lambda x:x[1])
            assert ranked[0][1] == 1
            new_score = Counter()
            doc_ids = set([d_id for d_id, _, _ in ranked[:top_k]])

            doc_len = dict()
            for doc_id in doc_ids:
                doc_len[doc_id] = len(collection[doc_id].split())
            query = queries[q_id]
            for doc_id in doc_ids:
                doc = collection[doc_id]
                new_score[doc_id] = get_bm25(query, doc, idf.df, N=len(collection), avdl=avdl)
            new_list = list(new_score.items())
            new_list.sort(key=lambda x:x[1], reverse=True)
            for doc_id in doc_ids:
                if doc_id not in new_score:
                    assert False
            rank_idx = 1
            for doc_id, score in new_list:
                fout.write("{} Q0 {} {} {} galago\n".format(q_id, doc_id, rank_idx, score))
                rank_idx += 1


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
            fout = cpath.open_pred_output("rerank_{}".format(exp_config.name))
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

        labels = controversy.load_clue303_label()
        docs = controversy.load_clue303_docs()

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

    def bert_lm_pos_neg(self, exp_config, data_loader, preload_id):
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
        good_id = encoder_unit.encoder.ft.convert_tokens_to_ids(["good"])[0]
        bad_id = encoder_unit.encoder.ft.convert_tokens_to_ids(["bad"])[0]

        print(good_id)
        print(bad_id)
        def encode(text):
            tokens_a = encoder_unit.encoder.encode(text)
            max_a_len = encoder_unit.max_seq - 2
            sub_tokens_a = tokens_a[:max_a_len]
            return encoder_unit.encode_inner(sub_tokens_a, [])

        def get_score(texts):
            def eval(runs):
                tprint("Packing batches (batch_size={})".format(batch_size))
                batches = get_batches_ex(runs, batch_size, 4)

                tprint("Runing neural network prediction (#batch={})".format(len(batches)))
                y_list = []
                x_list = []
                ticker = TimeEstimator(len(batches), sample_size=20)
                for batch in batches:
                    x0, x1, x2, y = batch
                    probs, = self.sess.run([task.log_probs, ],
                                           feed_dict={
                                               task.x_list[0]: x0,
                                               task.x_list[1]: x1,
                                               task.x_list[2]: x2,
                                               task.y: y,
                                           })
                    y_list.append(probs)
                    x_list.append(x0)
                    ticker.tick()
                xs = np.concatenate(x_list)
                ys = np.concatenate(y_list)
                return xs, ys

            runs = []
            for text in texts:
                encoded = encode(text)
                run = (encoded['input_ids'],
                       encoded['input_mask'],
                       encoded['segment_ids'],
                       0)
                runs.append(run)


            xs, ys = eval(runs)
            tprint("Completed GPU computations")
            return xs, ys

        texts = ["The worst thing Hillary Clinton has done has been to be a woman and a Democrat at a time when right-wing politics has replaced respect for truth and reasonable discourse with an ends-justifies-the means approach. Republicans aren’t the only ones who sling mud, but they’re better at it, and less encumbered by moral or ethical restraint. The various lies that came out of these nefarious efforts seem immortal—living on as conspiracy theories. \
        Some people who think they know better than the entire U.S. Justice department still think she should be in jail for things that multiple investigations found nothing even prosecutable in. Others try to find something nefarious about her being part of a foundation that has helped millions of people around the world and that has never given her a dime. Others accuse her of sending uranium to the Russians, which did not happen. \
        We can expect this swift-boating to happen with every future Democratic presidential candidate—especially when the Republican candidate has a lack of merit that makes party operatives feel they have to counterbalance with extra mud slinging. That’s how we ended up with George W. Bush and Donald Trump.",
                 ]
        def get_voca(idx):
            return encoder_unit.encoder.ft.convert_ids_to_tokens([idx])[0]

        xs, ys = get_score(texts)
        for i, text in enumerate(texts):
            print(text)
            for j in range(len(xs[i])):
                voca_scores = ys[i][j][0]
                s = voca_scores[good_id] - voca_scores[bad_id]
                term = get_voca(xs[i][j])
                print("{}({})".format(term, s), end=" ")
            print("")


    def bert_lm_test(self, exp_config, data_loader, preload_id):
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
        mask_id = encoder_unit.encoder.ft.convert_tokens_to_ids(["[MASK]"])[0]
        dummy_mask_id = encoder_unit.encoder.ft.convert_tokens_to_ids(["mask"])[0]

        print(mask_id)
        print(dummy_mask_id)

        def mask_encode(text):
            tokens_a = encoder_unit.encoder.encode(text)
            new_tokens = []
            for t in tokens_a:
                if t == dummy_mask_id:
                    new_tokens.append(mask_id)
                else:
                    new_tokens.append(t)

            tokens_a = new_tokens
            print(tokens_a)
            max_a_len = encoder_unit.max_seq - 2
            sub_tokens_a = tokens_a[:max_a_len]
            return encoder_unit.encode_inner(sub_tokens_a, [])

        def get_score(texts):
            def eval(runs):
                tprint("Packing batches (batch_size={})".format(batch_size))
                batches = get_batches_ex(runs, batch_size, 4)

                tprint("Runing neural network prediction (#batch={})".format(len(batches)))
                y_list = []
                x_list = []
                ticker = TimeEstimator(len(batches), sample_size=20)
                for batch in batches:
                    x0, x1, x2, y = batch
                    probs, = self.sess.run([task.log_probs, ],
                                           feed_dict={
                                               task.x_list[0]: x0,
                                               task.x_list[1]: x1,
                                               task.x_list[2]: x2,
                                               task.y: y,
                                           })
                    y_list.append(probs)
                    x_list.append(x0)
                    ticker.tick()
                xs = np.concatenate(x_list)
                ys = np.concatenate(y_list)
                return xs, ys


            runs = []
            for text in texts:
                encoded = mask_encode(text)
                run = (encoded['input_ids'],
                       encoded['input_mask'],
                       encoded['segment_ids'],
                       0)
                runs.append(run)


            xs, ys = eval(runs)
            tprint("Completed GPU computations")
            return xs, ys

        templage = "The capital of {} is mask,"
        country_list = [
            "China",
            "Japan",
            "Germany",
            "India",
            "United Kingdom",
            "France",
            "Italy",
            "Brazil",
            "Canada",
            "Korea, South",
            "Russia",
            "Spain",
            "Australia",
            "Mexico",
            "Indonesia",
        ]
        texts = []
        for country in country_list:
            texts.append(templage.format(country))

        def get_voca(idx):
            return encoder_unit.encoder.ft.convert_ids_to_tokens([idx])[0]

        xs, ys = get_score(texts)
        for i, text in enumerate(texts):
            print(text)
            for j in range(len(xs[i])):
                if xs[i][j] == mask_id:
                    voca_scores = ys[i][j][0]
                    ranked = np.argsort(voca_scores)
                    ranked = np.flip(ranked)
                    print(" candidate : ")
                    for k in range(20):
                        w_idx = ranked[k]
                        print(get_voca(w_idx), end=" / ")
                    print("")





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

        labels = controversy.load_clue303_label()
        docs = controversy.load_clue303_docs()

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


    def controversy_cie_train(self, exp_config, data_loader, preload_id):
        print("controversy_cie_train")
        task = token_regression(self.hparam, data_loader.voca_size)
        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()

        with tf.variable_scope("optimizer"):
            train_cls = self.get_train_op(task.loss)

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)
        self.load_model_white2(preload_id, exp_config.load_names)

        def batch2feed_dict(batch):
            x0, x1, x2, y = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y: y,
            }
            return feed_dict


        def train_fn(batch, step_i):
            loss_val, summary, prec, recall,  _ = self.sess.run([task.loss, self.merged, task.prec, task.recall, train_cls,
                                             ],
                                            feed_dict=batch2feed_dict(batch)
                                            )
            self.log.debug("Step {0} train loss={1:.04f} prec={2:.03f} recall={3:.03f}".format(step_i, loss_val,
                                                                                               prec, recall))
            self.train_writer.add_summary(summary, self.g_step)
            return loss_val, prec

        def save_fn():
            self.save_model(exp_config.name, 1)

        dev_batches = get_batches_ex(data_loader.get_dev_data(), self.hparam.batch_size, 4)
        train_batches = get_batches_ex(data_loader.get_train_data(), self.hparam.batch_size, 4)

        def valid_fn():
            loss_list = []
            prec_list = []
            recall_list = []
            for batch in dev_batches:
                loss_val, summary, prec, recall = self.sess.run([task.loss, self.merged, task.prec, task.recall],
                                                       feed_dict=batch2feed_dict(batch)
                                                       )
                loss_list.append(loss_val)
                prec_list.append(prec)
                recall_list.append(recall)

                self.test_writer.add_summary(summary, self.g_step)
            self.log.info("Validation : loss={0:.04f} prec={1:.04f} recall={2:.03f}".format(average(loss_list),
                                                                                            average(prec_list), average(recall_list)))

        valid_freq = 5
        for i in range(exp_config.num_epoch):
            loss, _ = epoch_runner(train_batches, train_fn,
                                   valid_fn, valid_freq,
                                   save_fn, self.save_interval)


    def controversy_cie_span_train(self, exp_config, data_loader, preload_id):
        print("controversy_cie_train")
        task = span_selection(self.hparam, data_loader.voca_size)
        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()

        with tf.variable_scope("optimizer"):
            train_cls = self.get_train_op(task.loss)

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)
        self.load_model_white2(preload_id, exp_config.load_names)

        def batch2feed_dict(batch):
            x0, x1, x2, b, e = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.begin: b,
                task.end: e,
            }
            return feed_dict

        def prec_recall(souts, batch):
            _, _, _, begin_bin, end_bin = batch

            def precision(pred, y):
                y_bool = y.astype(np.bool)
                tp = np.logical_and(y_bool, np.equal(pred, y))
                return np.count_nonzero(tp) / np.count_nonzero(pred)

            def recall(pred, y):
                y_bool = y.astype(np.bool)
                tp = np.logical_and(y_bool, np.equal(pred, y))
                return np.count_nonzero(tp) / np.count_nonzero(y)

            n_inst, seq_len = begin_bin.shape

            begin = np.argmax(begin_bin, axis=1)
            end = np.argmax(end_bin, axis=1)
            assert len(souts) == len(begin)
            pred_idx = np.argmax(souts, axis=1)


            n_tp = 0
            n_pred = 0
            n_gold = 0
            for i in range(n_inst):
                gold = np.zeros([seq_len], dtype=int)
                pred = np.zeros([seq_len], dtype=int)
                for j in range(begin[i], end[i]+1):
                    gold[j] = 1

                for j in range(pred_idx[i,0], pred_idx[i,1]):
                    pred[j] = 1

                tp = np.logical_and(gold, np.equal(pred, gold))
                n_tp += np.count_nonzero(tp)
                n_pred += np.count_nonzero(pred)
                n_gold += np.count_nonzero(gold)

            p = n_tp / n_pred if n_pred > 0 else 0
            r = n_tp / n_gold if n_gold > 0 else 0

            f1 = 2 * p * r / (p+r) if p+r > 0 else 0
            return p, r, f1

        def baselines():
            def random_method(batch):
                _, _, _, begin_bin, end_bin = batch
                n_inst, seq_len = begin_bin.shape
                sout = np.random.rand(n_inst, seq_len, 2)
                for i in range(n_inst):
                    for j in range(100, seq_len):
                        sout[i, j, 0] = 0
                        sout[i,j, 1] = 0
                return sout

            def cont10word(batch):
                input_ids, _, _, begin_bin, end_bin = batch
                n_inst, seq_len = begin_bin.shape

                controversy_wid = data_loader.encoder.encode("controversy")[0]
                sout = np.zeros((n_inst, seq_len, 2))
                for i in range(n_inst):
                    cnt = 0
                    select = False
                    for j in range(seq_len):
                        if input_ids[i,j] == controversy_wid :
                            select = True
                            sout[i, j, 0] = 1
                        elif select:
                            cnt += 1
                            if cnt == 10:
                                sout[i, j, 1] = 1
                return sout


            todo = [
                (random_method, "random"),
                (cont10word, "10word"),
            ]
            for method, method_name in todo:
                prec_list = []
                recall_list = []
                f1_list = []
                for batch in dev_batches:
                    sout = method(batch)
                    prec, recall, f1 = prec_recall(sout, batch)
                    prec_list.append(prec)
                    recall_list.append(recall)
                    f1_list.append(f1)
                self.log.info("{0} : prec={1:.04f} recall={2:.03f} f1={3:.03f}".format(method_name, average(prec_list),
                                                                                                average(recall_list),
                                                                                               average(f1_list)))

        def train_fn(batch, step_i):
            loss_val, summary, sout, _ = self.sess.run([task.loss, self.merged, task.sout, train_cls,
                                                                ],
                                                               feed_dict=batch2feed_dict(batch)
                                                               )

            prec, recall, f1 = prec_recall(sout, batch)
            self.log.debug("Step {0} train loss={1:.04f} prec={2:.03f} recall={3:.03f} f1={4:.03f}".format(step_i, loss_val,
                                                                                               prec, recall, f1))
            self.train_writer.add_summary(summary, self.g_step)
            return loss_val, 0

        def save_fn():
            self.save_model(exp_config.name, 1)

        dev_batches = get_batches_ex(data_loader.get_dev_data(), self.hparam.batch_size, 5)
        train_batches = get_batches_ex(data_loader.get_train_data(), self.hparam.batch_size, 5)
        baselines()
        def valid_fn():
            loss_list = []
            prec_list = []
            recall_list = []
            f1_list = []
            for batch in dev_batches:
                loss_val, sout, summary = self.sess.run([task.loss, task.sout, self.merged],
                                                                feed_dict=batch2feed_dict(batch)
                                                                )
                prec, recall, f1 = prec_recall(sout, batch)
                loss_list.append(loss_val)
                prec_list.append(prec)
                recall_list.append(recall)
                f1_list.append(f1)

                self.test_writer.add_summary(summary, self.g_step)
            self.log.info("Validation : loss={0:.04f} prec={1:.04f} recall={2:.03f} f1={3:.03f}".format(average(loss_list),
                                                                                            average(prec_list),
                                                                                            average(recall_list),
                                                                                           average(f1_list)))

        valid_freq = 5
        for i in range(exp_config.num_epoch):
            loss, _ = epoch_runner(train_batches, train_fn,
                                   valid_fn, valid_freq,
                                   save_fn, self.save_interval)


    def train_protest(self, exp_config, data_loader, preload_id):
        print("train_protest")
        task = transformer_binary(self.hparam, data_loader.voca_size, True)
        with tf.variable_scope("optimizer"):
            train_cls = self.get_train_op(task.loss)

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)
        self.load_model_white2(preload_id, exp_config.load_names)

        def batch2feed_dict(batch):
            x0,x1,x2, y  = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y: y,
            }
            return feed_dict

        def forward_run(inputs):
            batches = get_batches_ex(inputs, self.hparam.batch_size, 3)
            logit_list = []
            for batch in batches:
                x0, x1, x2 = batch
                logits,  = self.sess.run([task.sout, ],
                                               feed_dict={
                                                task.x_list[0]: x0,
                                                task.x_list[1]: x1,
                                                task.x_list[2]: x2,
                                               })
                logit_list.append(logits)
            return np.concatenate(logit_list)

        def train_fn(batch, step_i):
            loss_val, summary, _ = self.sess.run([task.loss, self.merged, train_cls,
                                             ],
                                            feed_dict=batch2feed_dict(batch)
                                            )
            self.log.debug("Step {0} train loss={1:.04f}".format(step_i, loss_val))
            self.train_writer.add_summary(summary, self.g_step)
            self.g_step += 1
            return loss_val, 0

        dev_batches = get_batches_ex(data_loader.get_dev_data(), self.hparam.batch_size, 4)
        train_batches = get_batches_ex(data_loader.get_train_data(), self.hparam.batch_size, 4)
        dev_X, dev_Y = load_protest.load_data("dev")
        print(len(dev_X))

        def get_ap(y_rank):
            y_rank.sort(key=lambda x:x[1], reverse=True)
            return AP(left(y_rank), dev_Y)



        def valid_fn():
            loss_list = []
            logit_list = []
            for batch in dev_batches:
                loss_val, logits, summary = self.sess.run([task.loss, task.logits, self.merged],
                                                  feed_dict=batch2feed_dict(batch)
                                                  )
                loss_list.append(loss_val)
                logit_list.append(logits)

                self.test_writer.add_summary(summary, self.g_step)
            self.log.info("Validation : loss={0:.04f}".format(average(loss_list)))

            logit_list = np.concatenate(logit_list)
            scores = logit_list[:,1]
            assert len(scores) == len(dev_X)

            y_rank = []
            gold_y = []
            for score, x in zip(list(scores), dev_X):
                name, doc = x
                y_rank.append((name, score))
                gold_y.append( dev_Y[name])
            print("AP : ", get_ap(y_rank))

            pred_y = np.argmax(logit_list, axis=1)
            all_result = eval_2label(pred_y, gold_y)
            true_result = all_result[1]
            f1 = sum([result['f1'] for result in all_result]) / 2
            print("F1", f1)
            print("F1(Protest)", true_result['f1'])

        def save_fn():
            self.save_model(exp_config.name, 1)

        valid_freq = 10
        # train_op
        print("Train epoch")
        num_epochs = exp_config.num_epoch
        for i_epoch in range(num_epochs):
            loss, _ = epoch_runner(train_batches, train_fn,
                                   valid_fn, valid_freq,
                                   save_fn, exp_config.save_interval)

        dev_data = []
        y_list = []
        for batch in dev_batches:
            x0, x1, x2, y = batch
            for i in range(len(batch)):
                dev_data.append((x0[i],x1[i], x2[i] ))
                y_list.append(y)

        y_list = np.concatenate(y_list)

        pred = np.argmax(forward_run(dev_data), axis=1)
        explains = explain_by_deletion(dev_data, "mismatch" ,forward_run)

        result = []
        for idx in range(len(dev_data)):
            input_ids, _, _ = dev_data[idx]
            for i in range(len(input_ids)):
                if input_ids[i] == SEP_ID:
                    idx_sep1 = i
                    break

            tokens = data_loader.encoder.decode_list(input_ids[1:idx_sep1])
            result.append((tokens, explains[idx][1:idx_sep1], pred[idx], y_list[idx]))

        visualize.visualize_single(result, "protest")


    def train_rte(self, exp_config, data_loader, preload_id):
        print("train_rte")
        tf.reset_default_graph()
        task = transformer_binary(self.hparam, data_loader.voca_size, True)
        with tf.variable_scope("optimizer"):
            train_cls = self.get_train_op(task.loss)

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)
        self.load_model_white2(preload_id, exp_config.load_names, False)

        def batch2feed_dict(batch):
            x0,x1,x2, y  = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y: y,
            }
            return feed_dict

        def forward_run(inputs):
            batches = get_batches_ex(inputs, self.hparam.batch_size, 3)
            logit_list = []
            for batch in batches:
                x0, x1, x2 = batch
                logits,  = self.sess.run([task.sout, ],
                                               feed_dict={
                                                task.x_list[0]: x0,
                                                task.x_list[1]: x1,
                                                task.x_list[2]: x2,
                                               })
                logit_list.append(logits)
            return np.concatenate(logit_list)

        def train_fn(batch, step_i):
            loss_val, summary, _ = self.sess.run([task.loss, self.merged, train_cls,
                                             ],
                                            feed_dict=batch2feed_dict(batch)
                                            )
            self.log.debug("Step {0} train loss={1:.04f}".format(step_i, loss_val))
            self.train_writer.add_summary(summary, self.g_step)
            self.g_step += 1
            return loss_val, 0

        dev_batches = get_batches_ex(data_loader.get_dev_data(), self.hparam.batch_size, 4)
        train_batches = get_batches_ex(data_loader.get_train_data(), self.hparam.batch_size, 4)

        def valid_fn():
            loss_list = []

            gold_y = []
            pred_y = []
            for batch in dev_batches[:100]:
                x0,x1,x2,y = batch
                logits, loss_val, summary = self.sess.run([task.logits, task.loss, self.merged],
                                                  feed_dict=batch2feed_dict(batch)
                                                  )
                loss_list.append(loss_val)
                gold_y.append(y)
                pred_y.append(np.argmax(logits, axis=1))

                self.test_writer.add_summary(summary, self.g_step)

            gold_y = np.concatenate(gold_y)
            pred_y = np.concatenate(pred_y)
            acc = np.sum(np.equal(pred_y, gold_y)) / len(pred_y)

            self.log.info("Validation : loss={0:.04f} acc={1:.04f}".format(average(loss_list), acc))


        def save_fn():
            return self.save_model(exp_config.name, 1)

        valid_freq = 1000
        # train_op
        print("Train epoch")
        num_epochs = exp_config.num_epoch
        for i_epoch in range(num_epochs):
            loss, _ = epoch_runner(train_batches, train_fn,
                                   valid_fn, valid_freq,
                                   save_fn, exp_config.save_interval)
        return save_fn()



    def eval_rte(self, exp_config, data_loader, model_path):
        print("eval_rte")
        tf.reset_default_graph()
        task = transformer_binary(self.hparam, data_loader.voca_size, False)
        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)

        loader = tf.train.Saver()
        loader.restore(self.sess, model_path)

        def batch2feed_dict(batch):
            x0,x1,x2, y  = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y: y,
            }
            return feed_dict

        dev_batches = get_batches_ex(data_loader.get_dev_data(), self.hparam.batch_size, 4)
        def valid_fn():
            loss_list = []

            gold_y = []
            pred_y = []
            for batch in dev_batches[:100]:
                x0,x1,x2,y = batch
                logits, loss_val = self.sess.run([task.logits, task.loss],
                                                  feed_dict=batch2feed_dict(batch)
                                                  )
                loss_list.append(loss_val)
                gold_y.append(y)
                pred_y.append(np.argmax(logits, axis=1))

            gold_y = np.concatenate(gold_y)
            pred_y = np.concatenate(pred_y)
            acc = np.sum(np.equal(pred_y, gold_y)) / len(pred_y)

            self.log.info("Validation : loss={0:.04f} acc={1:.04f}".format(average(loss_list), acc))
            return acc

        return valid_fn()

    def rte_visualize(self, exp_config, data_loader, preload_id):
        print("rte_visualize")
        task = transformer_binary(self.hparam, data_loader.voca_size, True)
        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.load_model_white2(preload_id, exp_config.load_names)
        dev_batches = get_batches_ex(data_loader.get_dev_data(), self.hparam.batch_size, 4)

        def batch2feed_dict(batch):
            x0,x1,x2, y  = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y: y,
            }
            return feed_dict

        target = task.logits[:,1] - task.logits[:,0]
        g = tf.gradients(target, task.model.embedding_output)
        gold_y = []
        pred_y = []

        result = []
        for batch in dev_batches[:1]:
            x0, x1, x2, y = batch
            logits, loss_val, g_out = self.sess.run([task.logits, task.loss, g],
                                                      feed_dict=batch2feed_dict(batch)
                                                      )
            predictions = np.argmax(logits, axis=1)
            gold_y.append(y)
            pred_y.append(predictions)
            g_out = g_out[0]
            g_token = np.sum(np.abs(g_out), axis=2)
            print(g_token.shape)
            for idx in range(len(y)):
                input_ids = x0[idx]
                ex1, ex2 = visualize.split_with_input_ids(g_token[idx], input_ids)
                enc1, enc2 = visualize.split_with_input_ids(input_ids, input_ids)
                tokens1 = data_loader.encoder.decode_list(enc1)
                tokens2 = data_loader.encoder.decode_list(enc2)
                result.append((ex1, ex2, tokens1, tokens2, predictions[idx], y[idx]))
        visualize.visualize_sent_pair(result, "rte")




    def train_wiki_contrv(self, exp_config, data_loader, preload_id):
        print("train_wiki_cont")
        task = transformer_binary(self.hparam, data_loader.voca_size, True)
        with tf.variable_scope("optimizer"):
            train_cls = self.get_train_op(task.loss)

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)
        self.load_model_white2(preload_id, exp_config.load_names)

        def batch2feed_dict(batch):
            x0,x1,x2, y  = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y: y,
            }
            return feed_dict

        def forward_run(inputs):
            batches = get_batches_ex(inputs, self.hparam.batch_size, 3)
            logit_list = []
            for batch in batches:
                x0, x1, x2 = batch
                logits,  = self.sess.run([task.sout, ],
                                               feed_dict={
                                                task.x_list[0]: x0,
                                                task.x_list[1]: x1,
                                                task.x_list[2]: x2,
                                               })
                logit_list.append(logits)
            return np.concatenate(logit_list)

        def train_fn(batch, step_i):
            loss_val, summary, _ = self.sess.run([task.loss, self.merged, train_cls,
                                             ],
                                            feed_dict=batch2feed_dict(batch)
                                            )
            self.log.debug("Step {0} train loss={1:.04f}".format(step_i, loss_val))
            self.train_writer.add_summary(summary, self.g_step)
            self.g_step += 1
            return loss_val, 0

        dev_batches = get_batches_ex(data_loader.get_dev_data(), self.hparam.batch_size, 4)
        train_batches = get_batches_ex(data_loader.get_train_data(), self.hparam.batch_size, 4)

        def valid_fn():
            loss_list = []
            logit_list = []
            acc_list = []
            for batch in dev_batches:
                loss_val, logits, summary, acc= self.sess.run([task.loss, task.logits, self.merged, task.acc],
                                                  feed_dict=batch2feed_dict(batch)
                                                  )
                loss_list.append(loss_val)
                logit_list.append(logits)
                acc_list.append(acc)

                self.test_writer.add_summary(summary, self.g_step)
            self.log.info("Validation : loss={0:.04f} acc={1:.04f}".format(average(loss_list), average(acc_list)))

            logit_list = np.concatenate(logit_list)
            scores = logit_list[:,1]

        def save_fn():
            self.save_model(exp_config.name, 1)

        # train_op
        print("Train epoch")
        num_epochs = exp_config.num_epoch
        for i_epoch in range(num_epochs):
            loss, _ = epoch_runner(train_batches, train_fn,
                                   valid_fn, exp_config.valid_freq,
                                   save_fn, exp_config.save_interval)
        save_fn()

    def train_cnn_wiki_contrv(self, exp_config):
        X, y, voca = get_wiki_doc()
        hp = self.hparam

        embedding_size= hp.embedding_size

        init_emb = cache.load_cache("init_emb_word2vec")
        word2idx = cache.load_cache(exp_config.input_name+".voca")
        if init_emb is None:
            w2v = word2vec.load_w2v()
            init_emb, word2idx = word2vec.build_index(voca, w2v, embedding_size)
            cache.save_to_pickle(init_emb, "init_emb_word2vec")
            cache.save_to_pickle(word2idx, exp_config.input_name + ".voca")



        dropout_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        cnn = CNN("controv", hp.seq_max, 2, [2,3,4], 128, init_emb, embedding_size, dropout_prob)
        input_text = tf.placeholder(tf.int32,
                                       shape=[None, hp.seq_max],
                                       name="comment_input")
        input_y = tf.placeholder(tf.int32, shape=[None, ],name="input_y")  # Controversy Label

        def batch2feed_dict(batch):
            x, y  = batch
            feed_dict = {
                input_text:x,
                input_y:y,
                dropout_prob:0.5,
            }
            return feed_dict
        sout = cnn.network(input_text)
        logits = cnn.logit
        #losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=sout, labels=input_y)
        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=sout, labels=tf.one_hot(input_y,2))
        loss = tf.reduce_mean(losses)
        train_cls = self.get_train_op(loss)
        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())

        data_loader = SimpleLoader(word2idx, X, y, hp.seq_max)


        train_batches = get_batches_ex(data_loader.get_train_data(), hp.batch_size, 2)
        dev_batches = get_batches_ex(data_loader.get_dev_data(), hp.batch_size, 2)

        def valid_fn():
            loss_list = []
            logit_list = []
            gold_y = []

            for batch in dev_batches:
                x, y = batch
                loss_val, logits = self.sess.run([loss, sout],
                                                  feed_dict=batch2feed_dict(batch)
                                                  )
                loss_list.append(loss_val)
                logit_list.append(logits)
                gold_y.append(y)

            logit_list = np.concatenate(logit_list)
            gold_y = np.concatenate(gold_y)
            pred_y = np.argmax(logit_list, axis=1)
            acc = np.sum(np.equal(pred_y, gold_y)) / len(pred_y)
            self.log.info("Dev : Loss = {} Acc = {}".format(average(loss_list), acc))

        def save_fn():
            return self.save_model(exp_config.name, 1)

        def train_fn(batch, step_i):
            loss_val, _ = self.sess.run([loss, train_cls,
                                             ],
                                            feed_dict=batch2feed_dict(batch)
                                            )
            self.log.debug("Step {0} train loss={1:.04f}".format(step_i, loss_val))
            self.g_step += 1
            return loss_val, 0

        valid_freq = 10
        # train_op
        print("Train epoch")
        num_epochs = exp_config.num_epoch
        for i_epoch in range(num_epochs):
            _, _ = epoch_runner(train_batches, train_fn,
                                   valid_fn, valid_freq,
                                   save_fn, exp_config.save_interval)

        steps = int(len(train_batches) * 0.2)
        loss, _ = sub_step_runner(train_batches, train_fn,
                                  valid_fn, valid_freq,
                                  save_fn, self.save_interval,
                                  steps=steps)


        save_fn()


    def eval_ukp(self, exp_config, data_loader, model_path):
        print("eval_ukp")
        tf.reset_default_graph()
        task = transformer_nli(self.hparam, exp_config.voca_size, 0, False)
        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)

        loader = tf.train.Saver()
        loader.restore(self.sess, model_path)

        def batch2feed_dict(batch):
            x0,x1,x2, y  = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y: y,
            }
            return feed_dict

        dev_batches = get_batches_ex(data_loader.get_dev_data(), self.hparam.batch_size, 4)

        idx_for = data_loader.labels.index("Argument_for")
        idx_against = data_loader.labels.index("Argument_against")

        def valid_fn():
            loss_list = []
            logit_list = []
            gold_y = []

            for batch in dev_batches:
                x0, x1, x2, y = batch
                loss_val, logits, summary = self.sess.run([task.loss, task.logits, self.merged],
                                                  feed_dict=batch2feed_dict(batch)
                                                  )
                loss_list.append(loss_val)
                logit_list.append(logits)
                gold_y.append(y)
                self.test_writer.add_summary(summary, self.g_step)
            self.log.info("Validation : loss={0:.04f}".format(average(loss_list)))

            logit_list = np.concatenate(logit_list)
            gold_y = np.concatenate(gold_y)
            pred_y = np.argmax(logit_list, axis=1)

            all_result = eval_3label(pred_y, gold_y)
            for_result = all_result[idx_for]
            against_result = all_result[idx_against]
            f1 = sum([result['f1'] for result in all_result]) / 3
            print("F1", f1)
            print("P_arg+", for_result['precision'])
            print("R_arg+", for_result['recall'])
            print("P_arg-", against_result['precision'])
            print("R_arg-", against_result['recall'])
            return all_result

        all_result = valid_fn()
        f1 = sum([result['f1'] for result in all_result]) / 3
        return f1


    def eval_ukp_on_shared(self, exp_config, data_loader, num_class_list, model_path):
        print("eval_ukp")
        tf.reset_default_graph()
        task = transformer_mt(self.hparam, exp_config.voca_size, num_class_list, False)

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)

        loader = tf.train.Saver()
        loader.restore(self.sess, model_path)

        def batch2feed_dict(batch_ex):
            task_idx, batch = batch_ex
            x0,x1,x2, y = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y[task_idx]: y,
            }
            return feed_dict

        dev_batches = get_batches_ex(data_loader.get_dev_data(), self.hparam.batch_size, 4)

        idx_for = data_loader.labels.index("Argument_for")
        idx_against = data_loader.labels.index("Argument_against")

        def valid_fn():
            loss_list = []
            logit_list = []
            gold_y = []
            t_idx = 0

            for batch in dev_batches:
                x0, x1, x2, y = batch
                loss_val, logits = self.sess.run([task.loss_list[t_idx], task.logit_list[t_idx]],
                                                  feed_dict=batch2feed_dict((t_idx, batch))
                                                  )
                loss_list.append(loss_val)
                logit_list.append(logits)
                gold_y.append(y)
            self.log.info("Validation : loss={0:.04f}".format(average(loss_list)))

            logit_list = np.concatenate(logit_list)
            gold_y = np.concatenate(gold_y)
            pred_y = np.argmax(logit_list, axis=1)

            all_result = eval_3label(pred_y, gold_y)
            for_result = all_result[idx_for]
            against_result = all_result[idx_against]
            f1 = sum([result['f1'] for result in all_result]) / 3
            print("F1", f1)
            print("P_arg+", for_result['precision'])
            print("R_arg+", for_result['recall'])
            print("P_arg-", against_result['precision'])
            print("R_arg-", against_result['recall'])
            return all_result

        all_result = valid_fn()
        f1 = sum([result['f1'] for result in all_result]) / 3
        return f1



    def train_ukp(self, exp_config, data_loader, preload_id):
        print("train_ukp")
        tf.reset_default_graph()
        task = transformer_nli(self.hparam, exp_config.voca_size, 0, True)
        with tf.variable_scope("optimizer"):
            train_cls = self.get_train_op(task.loss)

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)
        self.load_model_white2(preload_id, exp_config.load_names, False)

        def batch2feed_dict(batch):
            x0,x1,x2, y  = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y: y,
            }
            return feed_dict

        def forward_run(inputs):
            batches = get_batches_ex(inputs, self.hparam.batch_size, 3)
            logit_list = []
            for batch in batches:
                x0, x1, x2 = batch
                logits,  = self.sess.run([task.sout, ],
                                               feed_dict={
                                                task.x_list[0]: x0,
                                                task.x_list[1]: x1,
                                                task.x_list[2]: x2,
                                               })
                logit_list.append(logits)
            return np.concatenate(logit_list)

        def train_fn(batch, step_i):
            loss_val, summary, _ = self.sess.run([task.loss, self.merged, train_cls,
                                             ],
                                            feed_dict=batch2feed_dict(batch)
                                            )
            self.log.debug("Step {0} train loss={1:.04f}".format(step_i, loss_val))
            self.train_writer.add_summary(summary, self.g_step)
            self.g_step += 1
            return loss_val, 0

        dev_batches = get_batches_ex(data_loader.get_dev_data(), self.hparam.batch_size, 4)
        train_batches = get_batches_ex(data_loader.get_train_data(), self.hparam.batch_size, 4)

        idx_for = data_loader.labels.index("Argument_for")
        idx_against = data_loader.labels.index("Argument_against")


        def valid_fn():
            loss_list = []
            logit_list = []
            gold_y = []

            for batch in dev_batches:
                x0, x1, x2, y = batch
                loss_val, logits, summary = self.sess.run([task.loss, task.logits, self.merged],
                                                  feed_dict=batch2feed_dict(batch)
                                                  )
                loss_list.append(loss_val)
                logit_list.append(logits)
                gold_y.append(y)
                self.test_writer.add_summary(summary, self.g_step)


            logit_list = np.concatenate(logit_list)
            gold_y = np.concatenate(gold_y)
            pred_y = np.argmax(logit_list, axis=1)

            all_result = eval_3label(pred_y, gold_y)
            for_result = all_result[idx_for]
            against_result = all_result[idx_against]
            f1 = sum([result['f1'] for result in all_result]) / 3
            self.log.info("Validation : loss={0:.04f} F1={1:.03f}".format(average(loss_list), f1))
            return all_result

        def save_fn():
            return self.save_model(exp_config.name, 1)

        valid_freq = 10
        # train_op
        print("Train epoch")
        num_epochs = exp_config.num_epoch
        for i_epoch in range(num_epochs):
            loss, _ = epoch_runner(train_batches, train_fn,
                                   valid_fn, valid_freq,
                                   save_fn, exp_config.save_interval)

        path = save_fn()
        all_result = valid_fn()
        return path


    def train_shared(self, exp_config, shared_data_loader, num_class_list, preload_id):
        print("train_shared")
        tf.reset_default_graph()
        num_task = len(num_class_list)
        task = transformer_mt(self.hparam, exp_config.voca_size, num_class_list, True)
        with tf.variable_scope("optimizer"):
            train_cls_list = []
            for i in range(num_task):
                train_cls_list.append(self.get_train_op(task.loss_list[i]))

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        #self.merged_list = list([tf.summary.merge(task.summary_list[task_idx].values()) for task_idx in range(num_task)])
        self.setup_summary_writer(exp_config.name)
        self.load_model_white2(preload_id, exp_config.load_names, False)

        def batch2feed_dict(batch_ex):
            task_idx, batch = batch_ex
            x0,x1,x2, y = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y[task_idx]: y,
            }
            return feed_dict

        def train_fn(batch, step_i):
            task_idx, _ = batch
            loss_val, _ = self.sess.run([task.loss_list[task_idx],
                                                 # self.merged_list[task_idx],
                                                  train_cls_list[task_idx],
                                             ],
                                            feed_dict=batch2feed_dict(batch)
                                            )
            t_name = shared_data_loader.task_name_list[task_idx]
            self.log.debug("Step {0} train {1} loss={2:.04f}".format(step_i, t_name, loss_val))
            #self.train_writer.add_summary(summary, self.g_step)
            self.g_step += 1
            return loss_val, 0


        def print_arg_metric(pred_y, gold_y, loss_list):
            all_result = eval_3label(pred_y, gold_y)
            f1 = sum([result['f1'] for result in all_result]) / 3
            self.log.info("Validation Arg : loss={0:.04f} F1={1:.03f}".format(average(loss_list), f1))
            return all_result

        def print_nli_metric(pred_y, gold_y, loss_list):
            acc = get_acc(pred_y, gold_y)
            self.log.info("Validation NLI : loss={0:.04f} Acc={1:.03f}".format(average(loss_list), acc))
            return acc

        print_task_metric = [print_arg_metric, print_nli_metric]

        def valid_fn():
            def valid_fn_in(task_idx):
                loss_list = []
                logit_list = []
                gold_y = []
                for k in range(exp_config.num_dev_batches):
                    batch = shared_data_loader.get_dev_batch_from(task_idx)

                    loss_val, logits  = self.sess.run([task.loss_list[task_idx], task.logit_list[task_idx]],
                                                      feed_dict=batch2feed_dict(batch)
                                                      )
                    loss_list.append(loss_val)
                    logit_list.append(logits)

                    task_idx, raw_batch = batch
                    x0, x1, x2, y = raw_batch
                    gold_y.append(y)
                    #self.test_writer.add_summary(summary, self.g_step)

                logit_list = np.concatenate(logit_list)
                gold_y = np.concatenate(gold_y)
                pred_y = np.argmax(logit_list, axis=1)
                print_task_metric[task_idx](pred_y, gold_y, loss_list)

            for task_idx in range(num_task):
                valid_fn_in(task_idx)

        def save_fn():
            return self.save_model(exp_config.name, 1)

        last_save = time.time()
        # train_op
        print("Train epoch")
        for step_i in range(exp_config.num_steps):
            if valid_fn is not None:
                if step_i % exp_config.valid_freq == 0:
                    valid_fn()

            if save_fn is not None:
                if time.time() - last_save > exp_config.save_interval:
                    save_fn()
                    last_save = time.time()


            batch = shared_data_loader.get_train_batch()
            loss, acc = train_fn(batch, step_i)

        path = save_fn()
        all_result_list = valid_fn()
        return path


    def run_ukp_ex(self, exp_config, data_loader, load_run_name):
        print("run_ukp_ex")
        tf.reset_default_graph()
        task = transformer_nli(self.hparam, exp_config.voca_size, 5, True)
        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        load_path = get_model_full_path(load_run_name)
        self.load_model_white_fullpath(load_path, exp_config.load_names, False)

        def batch2feed_dict(batch):
            x0,x1,x2, y  = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y: y,
            }
            return feed_dict

        get_fn = partial(data_loader.get_next_predict_batch, self.hparam.batch_size)
        input_queue = QueueFeader(100, get_fn, False)

        try:
            while True:
                batch = input_queue.get()
                x0, x1, x2, _ = batch
                logits, conf_logit, c_soft = self.sess.run([task.logits, task.conf_logits, task.conf_softmax],
                                                      feed_dict=batch2feed_dict(batch))
                predictions = logits.argmax(axis=1)
                for i in range(len(logits)):
                    st, ed = data_loader.get_second_seg(x0[i])
                    if ed is None:
                        data_loader.append_dummy()
                    else:
                        e = x0[i][st:ed], conf_logit[i][st:ed]
                        data_loader.append_write(e)

        except StopIteration as e:
            data_loader.finish_write()
            return

    def train_ukp_ex(self, exp_config, data_loader, load_run_name, explain_tag):
        print("train_ukp_ex")
        tf.reset_default_graph()
        task = transformer_nli(self.hparam, exp_config.voca_size, 5, True)
        with tf.variable_scope("optimizer"):
            train_cls = self.get_train_op(task.loss)
            train_rl = self.get_train_op(task.rl_loss, name="rl")
        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)

        load_path = get_model_full_path(load_run_name)
        self.load_model_white_fullpath(load_path, exp_config.load_names, False)

        def batch2feed_dict(batch):
            x0,x1,x2, y  = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y: y,
            }
            return feed_dict

        def action_score(before_prob, after_prob, action):
            num_tag = np.count_nonzero(action)
            penalty = (num_tag - 3) * 0.1 if num_tag > 3 else 0
            if explain_tag == 'polarity':
                score = (before_prob[2] - before_prob[1]) - (after_prob[2] - after_prob[1])
            elif explain_tag == 'relevance':
                # Increase of neutral
                score = (before_prob[2] + before_prob[1]) - (after_prob[2] + after_prob[1])
                # ( 1 - before_prob[1] ) - (1 - after_prob[1]) = after_prob[1] - before_prob[1] = increase of neutral
            elif explain_tag == 'mismatch':
                score = before_prob[1] - after_prob[1]
            else:
                assert False
            score = score - penalty
            return score

        def forward_run(inputs):
            batches = get_batches_ex(inputs, self.hparam.batch_size, 3)
            logit_list = []
            for batch in batches:
                x0, x1, x2 = batch
                logits,  = self.sess.run([task.sout, ],
                                               feed_dict={
                                                task.x_list[0]: x0,
                                                task.x_list[1]: x1,
                                                task.x_list[2]: x2,
                                               })
                logit_list.append(logits)
            return np.concatenate(logit_list)

        def train_classification(batch, step_i):
            loss_val, summary, _ = self.sess.run([task.loss, self.merged, train_cls,
                                             ],
                                            feed_dict=batch2feed_dict(batch)
                                            )
            self.log.debug("Step {0} train loss={1:.04f}".format(step_i, loss_val))
            self.train_writer.add_summary(summary, self.g_step)
            self.g_step += 1
            return loss_val, 0



        explain_trainer = ExplainTrainer(
            forward_runs=forward_run,
            action_score=action_score,
            sess=self.sess,
            rl_loss=task.rl_loss,
            sout=task.sout,
            ex_logits=task.conf_logits,
            train_rl=train_rl,
            input_rf_mask =task.rf_mask,
            batch2feed_dict=batch2feed_dict,
            target_class_set=[2],
            hparam=self.hparam,
            log2=self.log2)

        def train_fn(batch, step_i):
            loss_val, acc = train_classification(batch, step_i)
            summary = tf.Summary()
            explain_trainer.train_batch(batch, summary)
            self.train_writer.add_summary(summary, self.g_step)
            self.train_writer.flush()

            return loss_val, acc

        dev_batches = get_batches_ex(data_loader.get_dev_data(), self.hparam.batch_size, 4)
        train_batches = get_batches_ex(data_loader.get_train_data(), self.hparam.batch_size, 4)

        idx_for = data_loader.labels.index("Argument_for")
        idx_against = data_loader.labels.index("Argument_against")


        def valid_fn():
            loss_list = []
            logit_list = []
            gold_y = []

            for batch in dev_batches:
                x0, x1, x2, y = batch
                loss_val, logits, summary = self.sess.run([task.loss, task.logits, self.merged],
                                                  feed_dict=batch2feed_dict(batch)
                                                  )
                loss_list.append(loss_val)
                logit_list.append(logits)
                gold_y.append(y)
                self.test_writer.add_summary(summary, self.g_step)


            logit_list = np.concatenate(logit_list)
            gold_y = np.concatenate(gold_y)
            pred_y = np.argmax(logit_list, axis=1)

            all_result = eval_3label(pred_y, gold_y)
            f1 = sum([result['f1'] for result in all_result]) / 3
            self.log.info("Validation : loss={0:.04f} F1={1:.03f}".format(average(loss_list), f1))
            return all_result

        def do_visualize():
            name = exp_config.name
            visualize.visualize_pair_run(name, self.sess, task.sout, task.conf_logits,
                                         dev_batches[:20], batch2feed_dict, data_loader.encoder)
        def save_fn():
            do_visualize()
            return self.save_model(exp_config.name, 1)

        valid_freq = 10
        # train_op
        print("Train epoch")
        do_visualize()
        num_epochs = exp_config.num_epoch
        for i_epoch in range(num_epochs):
            loss, _ = epoch_runner(train_batches, train_fn,
                                   valid_fn, valid_freq,
                                   save_fn, exp_config.save_interval)

        path = save_fn()
        all_result = valid_fn()

        return path

    def train_ukp_weight(self, exp_config, data_loader, preload_id):
        print("train_ukp_weight")
        tf.reset_default_graph()
        task = transformer_weight(self.hparam, exp_config.voca_size, True)
        with tf.variable_scope("optimizer"):
            train_cls = self.get_train_op(task.loss)

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)
        self.load_model_white2(preload_id, exp_config.load_names)

        def batch2feed_dict(batch):
            x0,x1,x2, y  = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y: y,
            }
            return feed_dict

        def forward_run(inputs):
            batches = get_batches_ex(inputs, self.hparam.batch_size, 3)
            logit_list = []
            for batch in batches:
                x0, x1, x2 = batch
                logits,  = self.sess.run([task.sout, ],
                                               feed_dict={
                                                task.x_list[0]: x0,
                                                task.x_list[1]: x1,
                                                task.x_list[2]: x2,
                                               })
                logit_list.append(logits)
            return np.concatenate(logit_list)

        def train_fn(batch, step_i):
            loss_val, summary, _ = self.sess.run([task.loss, self.merged, train_cls,
                                             ],
                                            feed_dict=batch2feed_dict(batch)
                                            )
            self.log.debug("Step {0} train loss={1:.04f}".format(step_i, loss_val))
            self.train_writer.add_summary(summary, self.g_step)
            self.g_step += 1
            return loss_val, 0

        dev_batches = get_batches_ex(data_loader.get_dev_data(), self.hparam.batch_size, 4)
        train_batches = get_batches_ex(data_loader.get_train_data(), self.hparam.batch_size, 4)

        idx_for = data_loader.labels.index("Argument_for")
        idx_against = data_loader.labels.index("Argument_against")


        def valid_fn():
            loss_list = []
            logit_list = []
            gold_y = []

            for batch in dev_batches:
                x0, x1, x2, y = batch
                loss_val, logits, summary = self.sess.run([task.loss, task.logits, self.merged],
                                                  feed_dict=batch2feed_dict(batch)
                                                  )
                loss_list.append(loss_val)
                logit_list.append(logits)
                gold_y.append(y)
                self.test_writer.add_summary(summary, self.g_step)
            self.log.info("Validation : loss={0:.04f}".format(average(loss_list)))

            logit_list = np.concatenate(logit_list)
            gold_y = np.concatenate(gold_y)
            pred_y = np.argmax(logit_list, axis=1)

            all_result = eval_3label(pred_y, gold_y)
            for_result = all_result[idx_for]
            against_result = all_result[idx_against]
            f1 = sum([result['f1'] for result in all_result]) / 3
            print("F1", f1)
            print("P_arg+", for_result['precision'])
            print("R_arg+", for_result['recall'])
            print("P_arg-", against_result['precision'])
            print("R_arg-", against_result['recall'])
            return all_result

        def save_fn():
            self.save_model(exp_config.name, 1)

        valid_freq = 10
        # train_op
        print("Train epoch")
        num_epochs = exp_config.num_epoch
        for i_epoch in range(num_epochs):
            loss, _ = epoch_runner(train_batches, train_fn,
                                   valid_fn, valid_freq,
                                   save_fn, exp_config.save_interval)

        all_result = valid_fn()
        f1 = sum([result['f1'] for result in all_result]) / 3
        return f1

    def train_ukp_concat(self, exp_config, data_loader, preload_id1, preload_id2):
        print("train_ukp_concat")
        tf.reset_default_graph()
        task = transformer_concat(self.hparam, exp_config.voca_size, 0, True)
        with tf.variable_scope("optimizer"):
            train_cls = self.get_train_op(task.loss)

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)
        self.load_concat_model([preload_id1, preload_id2], exp_config.load_names, ["part1", "part2"])

        def batch2feed_dict(batch):
            x0,x1,x2, y  = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y: y,
            }
            return feed_dict

        def forward_run(inputs):
            batches = get_batches_ex(inputs, self.hparam.batch_size, 3)
            logit_list = []
            for batch in batches:
                x0, x1, x2 = batch
                logits,  = self.sess.run([task.sout, ],
                                               feed_dict={
                                                task.x_list[0]: x0,
                                                task.x_list[1]: x1,
                                                task.x_list[2]: x2,
                                               })
                logit_list.append(logits)
            return np.concatenate(logit_list)

        def train_fn(batch, step_i):
            loss_val, summary, _ = self.sess.run([task.loss, self.merged, train_cls,
                                             ],
                                            feed_dict=batch2feed_dict(batch)
                                            )
            self.log.debug("Step {0} train loss={1:.04f}".format(step_i, loss_val))
            self.train_writer.add_summary(summary, self.g_step)
            self.g_step += 1
            return loss_val, 0

        dev_batches = get_batches_ex(data_loader.get_dev_data(), self.hparam.batch_size, 4)
        train_batches = get_batches_ex(data_loader.get_train_data(), self.hparam.batch_size, 4)

        idx_for = data_loader.labels.index("Argument_for")
        idx_against = data_loader.labels.index("Argument_against")


        def valid_fn():
            loss_list = []
            logit_list = []
            gold_y = []

            for batch in dev_batches:
                x0, x1, x2, y = batch
                loss_val, logits, summary = self.sess.run([task.loss, task.logits, self.merged],
                                                  feed_dict=batch2feed_dict(batch)
                                                  )
                loss_list.append(loss_val)
                logit_list.append(logits)
                gold_y.append(y)
                self.test_writer.add_summary(summary, self.g_step)
            self.log.info("Validation : loss={0:.04f}".format(average(loss_list)))

            logit_list = np.concatenate(logit_list)
            gold_y = np.concatenate(gold_y)
            pred_y = np.argmax(logit_list, axis=1)

            all_result = eval_3label(pred_y, gold_y)
            for_result = all_result[idx_for]
            against_result = all_result[idx_against]
            f1 = sum([result['f1'] for result in all_result]) / 3
            print("F1", f1)
            print("P_arg+", for_result['precision'])
            print("R_arg+", for_result['recall'])
            print("P_arg-", against_result['precision'])
            print("R_arg-", against_result['recall'])
            return all_result

        def save_fn():
            self.save_model(exp_config.name, 1)

        valid_freq = 10
        # train_op
        print("Train epoch")
        num_epochs = exp_config.num_epoch
        for i_epoch in range(num_epochs):
            loss, _ = epoch_runner(train_batches, train_fn,
                                   valid_fn, valid_freq,
                                   save_fn, exp_config.save_interval)

        all_result = valid_fn()
        f1 = sum([result['f1'] for result in all_result]) / 3
        return f1

    def train_ukp_paired(self, exp_config, data_loader, preload_id):
        tprint("train_ukp_paired")
        num_epochs = exp_config.num_epoch

        tf.reset_default_graph()
        task = transformer_paired(self.hparam, exp_config.voca_size, 0, True)
        with tf.variable_scope("optimizer"):
            train_cls = self.get_train_op(task.loss_paired)

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)
        tprint("Loading saved model")
        self.load_model_white2(preload_id, exp_config.load_names, False)

        def batch2feed_dict(batch):
            x0,x1,x2,y1,y2,f_loc1, f_loc2,  = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y1: y1,
                task.y2: y2,
                task.f_loc1: f_loc1,
                task.f_loc2: f_loc2,
            }
            return feed_dict

        def train_fn(batch, step_i):
            loss_val, _ = self.sess.run([task.loss_paired, train_cls],
                                            feed_dict=batch2feed_dict(batch)
                                            )
            self.log.debug("Step {0} train loss={1:.04f}".format(step_i, loss_val))
            self.g_step += 1
            return loss_val, 0

        tprint("Loading dev data")
        dev_batches = get_batches_ex(data_loader.get_dev_data(), self.hparam.batch_size, 7)
        tprint("Loading train data")
        train_batches = get_batches_ex(data_loader.get_repeated_train(num_epochs), self.hparam.batch_size, 7)

        idx_for = data_loader.labels.index("Argument_for")
        idx_against = data_loader.labels.index("Argument_against")


        def valid_fn():
            loss_list = []
            logit_list = []
            gold_y = []

            for batch in dev_batches:
                x0, x1, x2, y1, y2,_,_ = batch
                loss_val, logits = self.sess.run([task.loss_paired, task.logits1],
                                                  feed_dict=batch2feed_dict(batch)
                                                  )
                loss_list.append(loss_val)
                logit_list.append(logits)
                gold_y.append(y1)
                #self.test_writer.add_summary(summary, self.g_step)
            self.log.info("Validation : loss={0:.04f}".format(average(loss_list)))

            logit_list = np.concatenate(logit_list)
            gold_y = np.concatenate(gold_y)
            pred_y = np.argmax(logit_list, axis=1)

            all_result = eval_3label(pred_y, gold_y)
            for_result = all_result[idx_for]
            against_result = all_result[idx_against]
            f1 = sum([result['f1'] for result in all_result]) / 3
            print("F1", f1)
            print("P_arg+", for_result['precision'])
            print("R_arg+", for_result['recall'])
            print("P_arg-", against_result['precision'])
            print("R_arg-", against_result['recall'])
            return all_result

        def save_fn():
            self.save_model(exp_config.name, 1)

        valid_freq = 10
        # train_op
        print("Train epoch")
        for i_epoch in range(num_epochs):
            loss, _ = epoch_runner(train_batches, train_fn,
                                   valid_fn, valid_freq,
                                   save_fn, exp_config.save_interval)

        all_result = valid_fn()
        f1 = sum([result['f1'] for result in all_result]) / 3
        return f1


    def failure_ukp(self, exp_config, topic):
        tf.reset_default_graph()
        task = transformer_nli(self.hparam, exp_config.voca_size, 0, True)
        data_loader = ukp.BertDataLoader(topic, True, self.hparam.seq_max, "bert_voca.txt", "is_good")

        def load_available():
            run_dir = os.path.join(self.model_dir, 'runs')
            save_dir = os.path.join(run_dir, exp_config.name)
            model_id = None
            for (dirpath, dirnames, filenames) in os.walk(save_dir):
                for filename in filenames:
                    if ".meta" in filename:
                        print(filename)
                        model_id = filename[:-5]

            print(model_id)
            assert model_id is not None
            self.load_model_white2((exp_config.name, model_id), ['bert', "cls_dense"], False)

        def batch2feed_dict(batch):
            x0,x1,x2, y  = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y: y,
            }
            return feed_dict

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)
        load_available()

        dev_batches = get_batches_ex(data_loader.get_dev_data(), self.hparam.batch_size, 4)

        idx_for = data_loader.labels.index("Argument_for")
        idx_against = data_loader.labels.index("Argument_against")

        def valid_fn():
            loss_list = []
            logit_list = []
            gold_y = []

            for batch in dev_batches:
                x0, x1, x2, y = batch
                loss_val, logits, summary = self.sess.run([task.loss, task.logits, self.merged],
                                                  feed_dict=batch2feed_dict(batch)
                                                  )
                loss_list.append(loss_val)
                logit_list.append(logits)
                gold_y.append(y)
                self.test_writer.add_summary(summary, self.g_step)
            self.log.info("Validation : loss={0:.04f}".format(average(loss_list)))

            logit_list = np.concatenate(logit_list)
            gold_y = np.concatenate(gold_y)
            pred_y = np.argmax(logit_list, axis=1)

            all_result = eval_3label(pred_y, gold_y)
            for_result = all_result[idx_for]
            against_result = all_result[idx_against]
            f1 = sum([result['f1'] for result in all_result]) / 3
            print("F1", f1)
            print("P_arg+", for_result['precision'])
            print("R_arg+", for_result['recall'])
            print("P_arg-", against_result['precision'])
            print("R_arg-", against_result['recall'])
            return all_result

        valid_fn()
        logit_list = []
        for batch in dev_batches:
            loss_val, logits, summary = self.sess.run([task.loss, task.logits, self.merged],
                                                      feed_dict=batch2feed_dict(batch)
                                                      )
            logit_list.append(logits)
        logit_list = np.concatenate(logit_list)
        pred_list = np.argmax(logit_list, axis=1)

        entries = data_loader.all_data[topic]
        val_entries = list([e for e in entries if e['set'] == 'val'])
        assert len(pred_list) == len(val_entries)
        confusion = Counter()
        print("[{}]".format(topic))
        for pred, entry in zip(pred_list, val_entries):
            #print(entry['sentence'])
            result = "wrong" if pred != data_loader.annotation2label(entry['annotation']) else "correct"
            gold_y = data_loader.labels.index(entry['annotation'])
            #print(result, "{}({})".format(pred, gold_y))
            confusion[(pred, gold_y)] += 1


        for pred_i in range(3):
            print(pred_i, end = "\t")
            for gold_i in range(3):
                print(confusion[(pred_i, gold_i)], end="\t")
            print("")



    def train_agree(self, exp_config, data_loader, preload_id):
        print("train_agree")
        tf.reset_default_graph()
        task = transformer_weight(self.hparam, exp_config.voca_size, True)
        with tf.variable_scope("optimizer"):
            train_cls = self.get_train_op(task.loss)

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)
        self.load_model_white2(preload_id, exp_config.load_names)

        def batch2feed_dict(batch):
            x0,x1,x2, y  = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y: y,
            }
            return feed_dict

        def forward_run(inputs):
            batches = get_batches_ex(inputs, self.hparam.batch_size, 3)
            logit_list = []
            for batch in batches:
                x0, x1, x2 = batch
                logits,  = self.sess.run([task.sout, ],
                                               feed_dict={
                                                task.x_list[0]: x0,
                                                task.x_list[1]: x1,
                                                task.x_list[2]: x2,
                                               })
                logit_list.append(logits)
            return np.concatenate(logit_list)

        def train_fn(batch, step_i):
            loss_val, summary, _ = self.sess.run([task.loss, self.merged, train_cls,
                                             ],
                                            feed_dict=batch2feed_dict(batch)
                                            )
            self.log.debug("Step {0} train loss={1:.04f}".format(step_i, loss_val))
            self.train_writer.add_summary(summary, self.g_step)
            self.g_step += 1
            return loss_val, 0

        dev_batches = get_batches_ex(data_loader.get_dev_data(), self.hparam.batch_size, 4)
        train_batches = get_batches_ex(data_loader.get_train_data(), self.hparam.batch_size, 4)

        def valid_fn():
            loss_list = []
            acc_list = []
            for batch in dev_batches:
                x0, x1, x2, y = batch
                loss_val, logits, acc, summary = self.sess.run([task.loss, task.logits, task.acc, self.merged],
                                                  feed_dict=batch2feed_dict(batch)
                                                  )
                loss_list.append(loss_val)
                self.test_writer.add_summary(summary, self.g_step)
                acc_list.append(acc)
            self.log.info("Validation : loss={0:.04f}".format(average(loss_list)))
            self.log.info("Validation : acc={0:.04f}".format(average(acc_list)))


        def save_fn():
            return self.save_model(exp_config.name, 1)

        valid_freq = 10
        # train_op
        print("Train epoch")
        num_epochs = exp_config.num_epoch
        for i_epoch in range(num_epochs):
            loss, _ = epoch_runner(train_batches, train_fn,
                                   valid_fn, valid_freq,
                                   save_fn, exp_config.save_interval)

        return save_fn()


    def nli_as_stance(self, exp_config, topic, data_loader):
        print("nli_as_stance")
        tf.reset_default_graph()
        task = transformer_nli(self.hparam, exp_config.voca_size, 0, True)

        def load_available():
            run_dir = os.path.join(self.model_dir, 'runs')
            save_dir = os.path.join(run_dir, exp_config.preload_name)
            model_id = None
            for (dirpath, dirnames, filenames) in os.walk(save_dir):
                for filename in filenames:
                    if ".meta" in filename:
                        print(filename)
                        model_id = filename[:-5]

            print(model_id)
            assert model_id is not None
            self.load_model_white2((exp_config.load_name, model_id), ['bert', "cls_dense"], False)

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)
        self.load_model_white2(exp_config.preload_id, exp_config.load_names, True)
        dev_batches = get_batches_ex(data_loader.get_dev_data(), self.hparam.batch_size, 4)

        idx_for = data_loader.labels.index("Argument_for")
        idx_against = data_loader.labels.index("Argument_against")

        def batch2feed_dict(batch):
            x0, x1, x2, y = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y: y,
            }
            return feed_dict

        def logit_reorder(logits):
            new_v = [logits[:,1], logits[:,2], logits[:,0]]
            return np.stack(new_v, axis=1)

        def valid_fn():
            loss_list = []
            logit_list = []
            gold_y = []

            for batch in dev_batches:
                x0, x1, x2, y = batch
                loss_val, logits, summary = self.sess.run([task.loss, task.logits, self.merged],
                                                  feed_dict=batch2feed_dict(batch)
                                                  )
                logits = logit_reorder(logits)
                loss_list.append(loss_val)
                logit_list.append(logits)
                gold_y.append(y)
                self.test_writer.add_summary(summary, self.g_step)


            logit_list = np.concatenate(logit_list)
            pred_list = np.argmax(logit_list, axis=1)
            gold_y = np.concatenate(gold_y)
            pred_y = np.argmax(logit_list, axis=1)

            all_result = eval_3label(pred_y, gold_y)
            for_result = all_result[idx_for]
            against_result = all_result[idx_against]
            f1 = sum([result['f1'] for result in all_result]) / 3

            entries = data_loader.all_data[topic]
            val_entries = list([e for e in entries if e['set'] == 'val'])

            for pred, entry in zip(pred_list, val_entries):
                print(entry['sentence'])
                result = "wrong" if pred != data_loader.annotation2label(entry['annotation']) else "correct"
                gold_y = data_loader.labels.index(entry['annotation'])
                print(result, "{}({})".format(pred, gold_y))

            self.log.info("Validation : loss={0:.04f} F1={1:.03f}".format(average(loss_list), f1))
            self.log.info("For Prec={0:.03f} Recall={1:.03f}".format(for_result['precision'], for_result['recall']))
            self.log.info("Against Prec={0:.03f} Recall={1:.03f}".format(against_result['precision'], against_result['recall']))


            return all_result
        valid_fn()

    def pseudo_stance_ukp(self, exp_config, topic, feedback_data):
        tf.reset_default_graph()
        task = transformer_nli(self.hparam, exp_config.voca_size, 0, True)
        with tf.variable_scope("optimizer"):
            train_cls = self.get_train_op(task.loss)

        def load_available():
            run_dir = os.path.join(self.model_dir, 'runs')
            save_dir = os.path.join(run_dir, exp_config.preload_name)
            model_id = None
            for (dirpath, dirnames, filenames) in os.walk(save_dir):
                for filename in filenames:
                    if ".meta" in filename:
                        print(filename)
                        model_id = filename[:-5]

            print(model_id)
            assert model_id is not None
            self.load_model_white2((exp_config.load_name, model_id), ['bert', "cls_dense"], False)

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)
        train_batches = get_batches_ex(feedback_data.get_train_data(), self.hparam.batch_size, 4)

        data_loader = ukp.BertDataLoader(topic, True, self.hparam.seq_max, "bert_voca.txt", "only_topic_word")
        dev_batches = get_batches_ex(data_loader.get_dev_data(), self.hparam.batch_size, 4)
        #load_available()
        self.load_model_white2(exp_config.preload_id, exp_config.load_names, False)

        def batch2feed_dict(batch):
            x0, x1, x2, y = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y: y,
            }
            return feed_dict

        def train_fn(batch, step_i):
            loss_val, summary, _ = self.sess.run([task.loss, self.merged, train_cls,
                                             ],
                                            feed_dict=batch2feed_dict(batch)
                                            )
            self.log.debug("Step {0} train loss={1:.04f}".format(step_i, loss_val))
            self.train_writer.add_summary(summary, self.g_step)
            self.g_step += 1
            return loss_val, 0

        idx_for = data_loader.labels.index("Argument_for")
        idx_against = data_loader.labels.index("Argument_against")


        def valid_fn():
            loss_list = []
            logit_list = []
            gold_y = []

            for batch in dev_batches:
                x0, x1, x2, y = batch
                loss_val, logits, summary = self.sess.run([task.loss, task.logits, self.merged],
                                                  feed_dict=batch2feed_dict(batch)
                                                  )
                loss_list.append(loss_val)
                logit_list.append(logits)
                gold_y.append(y)
                self.test_writer.add_summary(summary, self.g_step)


            logit_list = np.concatenate(logit_list)
            gold_y = np.concatenate(gold_y)
            pred_y = np.argmax(logit_list, axis=1)

            all_result = eval_3label(pred_y, gold_y)
            for_result = all_result[idx_for]
            against_result = all_result[idx_against]
            f1 = sum([result['f1'] for result in all_result]) / 3
            self.log.info("Validation : loss={0:.04f} F1={1:.03f}".format(average(loss_list), f1))
            return all_result

        def save_fn():
            return self.save_model(exp_config.name, 1)

        valid_freq = 10
        for i_epoch in range(exp_config.num_epochs):
            loss, _ = epoch_runner(train_batches, train_fn,
                                   valid_fn, valid_freq,
                                   save_fn, exp_config.save_interval)



    def pseudo_stance_ukp_vector(self, exp_config, topic, feedback_data):
        tf.reset_default_graph()
        task = transformer_nli_vector(self.hparam, exp_config.voca_size, True)
        with tf.variable_scope("optimizer"):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.hparam.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
            train_op = optimizer.minimize(task.loss, global_step=self.global_step, var_list=[task.fixed_embedding])

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)
        train_batches = get_batches_ex(feedback_data.get_train_data(), self.hparam.batch_size, 4)

        data_loader = ukp.BertDataLoader(topic, True, self.hparam.seq_max, "bert_voca.txt", "only_topic_word")
        dev_batches = get_batches_ex(data_loader.get_dev_data(), self.hparam.batch_size, 4)
        #load_available()
        self.load_model_white2(exp_config.preload_id, exp_config.load_names, True)

        def batch2feed_dict(batch):
            x0, x1, x2, y = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y: y,
            }
            return feed_dict

        def train_fn(batch, step_i):
            embeddings, attention = self.sess.run([task.encoded_embedding_out, task.attention_mask_out
                                             ],
                                            feed_dict=batch2feed_dict(batch)
                                            )
            x0, x1, x2, y = batch
            loss_val, summary, _ = self.sess.run([task.loss, self.merged, train_op,
                                             ],
                                            feed_dict={
                                                task.encoded_embedding_in:embeddings,
                                                task.attention_mask_in:attention,
                                                task.y: y,
                                            }
                                            )
            self.log.debug("Step {0} train loss={1:.04f}".format(step_i, loss_val))
            self.train_writer.add_summary(summary, self.g_step)
            self.g_step += 1
            return loss_val, 0

        idx_for = data_loader.labels.index("Argument_for")
        idx_against = data_loader.labels.index("Argument_against")


        def valid_fn():
            loss_list = []
            logit_list = []
            gold_y = []

            for batch in dev_batches:
                x0, x1, x2, y = batch
                embeddings, attention = self.sess.run([task.encoded_embedding_out, task.attention_mask_out
                                            ],
                                           feed_dict=batch2feed_dict(batch)
                                           )
                loss_val, logits, summary = self.sess.run([task.loss, task.logits, self.merged,
                                                      ],
                                                     feed_dict={
                                                         task.encoded_embedding_in: embeddings,
                                                         task.attention_mask_in: attention,
                                                         task.y: y,
                                                     }
                                                     )

                loss_list.append(loss_val)
                logit_list.append(logits)
                gold_y.append(y)
                self.test_writer.add_summary(summary, self.g_step)


            logit_list = np.concatenate(logit_list)
            gold_y = np.concatenate(gold_y)
            pred_y = np.argmax(logit_list, axis=1)

            all_result = eval_3label(pred_y, gold_y)
            for_result = all_result[idx_for]
            against_result = all_result[idx_against]
            f1 = sum([result['f1'] for result in all_result]) / 3
            self.log.info("Validation : loss={0:.04f} F1={1:.03f}".format(average(loss_list), f1))
            return all_result

        def save_fn():
            return self.save_model(exp_config.name, 1)

        valid_freq = 10
        for i_epoch in range(exp_config.num_epochs):
            loss, _ = epoch_runner(train_batches, train_fn,
                                   valid_fn, valid_freq,
                                   save_fn, exp_config.save_interval)



    def train_topic_vector(self, exp_config, topic, data_loader):
        tf.reset_default_graph()
        task = transformer_nli_vector(self.hparam, exp_config.voca_size, True)
        with tf.variable_scope("optimizer"):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.hparam.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
            train_op = optimizer.minimize(task.loss, global_step=self.global_step, var_list=[task.fixed_embedding])

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)
        train_batches = get_batches_ex(data_loader.get_hidden_train_data(), self.hparam.batch_size, 4)
        dev_batches = get_batches_ex(data_loader.get_dev_data(), self.hparam.batch_size, 4)


        def load_available():
            run_dir = os.path.join(self.model_dir, 'runs')
            save_dir = os.path.join(run_dir, exp_config.preload_name)
            model_id = None
            for (dirpath, dirnames, filenames) in os.walk(save_dir):
                for filename in filenames:
                    if ".meta" in filename:
                        print(filename)
                        model_id = filename[:-5]

            print(model_id)
            assert model_id is not None
            self.load_model_white2((exp_config.preload_name, model_id), ['bert', "cls_dense"], False)
        load_available()

        def batch2feed_dict(batch):
            x0, x1, x2, y = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y: y,
            }
            return feed_dict

        def reform_attention(attention):
            fixed_v = self.hparam.fixed_v
            b_size, _, _ = attention.shape
            attention[:, -fixed_v:, :] = 1
            attention[:, :, -fixed_v:] = 1
            return attention

        def train_fn(batch, step_i):
            embeddings, attention = self.sess.run([task.encoded_embedding_out, task.attention_mask_out
                                             ],
                                            feed_dict=batch2feed_dict(batch)
                                            )
            #attention = reform_attention(attention)
            x0, x1, x2, y = batch
            loss_val, summary, _ = self.sess.run([task.loss, self.merged, train_op,
                                             ],
                                            feed_dict={
                                                task.encoded_embedding_in:embeddings,
                                                task.attention_mask_in:attention,
                                                task.y: y,
                                            }
                                            )
            self.log.debug("Step {0} train loss={1:.04f}".format(step_i, loss_val))
            self.train_writer.add_summary(summary, self.g_step)
            self.g_step += 1
            return loss_val, 0

        idx_for = data_loader.labels.index("Argument_for")
        idx_against = data_loader.labels.index("Argument_against")
        entries = data_loader.all_data[topic]
        val_entries = list([e for e in entries if e['set'] == 'val'])

        def valid_fn():
            loss_list = []
            logit_list = []
            gold_y = []

            for batch in dev_batches:
                x0, x1, x2, y = batch
                embeddings, attention = self.sess.run([task.encoded_embedding_out, task.attention_mask_out
                                            ],
                                           feed_dict=batch2feed_dict(batch)
                                           )
                #attention = reform_attention(attention)
                loss_val, logits, summary = self.sess.run([task.loss, task.logits, self.merged,
                                                      ],
                                                     feed_dict={
                                                         task.encoded_embedding_in: embeddings,
                                                         task.attention_mask_in: attention,
                                                         task.y: y,
                                                     }
                                                     )

                loss_list.append(loss_val)
                logit_list.append(logits)
                gold_y.append(y)
                self.test_writer.add_summary(summary, self.g_step)


            logit_list = np.concatenate(logit_list)
            gold_y = np.concatenate(gold_y)
            pred_y = np.argmax(logit_list, axis=1)

            all_result = eval_3label(pred_y, gold_y)
            for_result = all_result[idx_for]
            against_result = all_result[idx_against]
            f1 = sum([result['f1'] for result in all_result]) / 3
            self.log.info("Validation : loss={0:.04f} F1={1:.03f}".format(average(loss_list), f1))
            self.log.info("P+={0:.03f} R+={1:.03f}".format(for_result['precision'], for_result['recall']))
            self.log.info("P-={0:.03f} R-={1:.03f}".format(against_result['precision'], against_result['recall']))


            for pred, entry in zip(pred_y, val_entries):
                if pred > 0:
                    gold_y = data_loader.labels.index(entry['annotation'])
                    result = "wrong" if pred != data_loader.annotation2label(entry['annotation']) else "correct"
                    print(result, "{}({})".format(pred, gold_y), end="\t")
                    print(entry['sentence'])


            return all_result


        def save_fn():
            return self.save_model(exp_config.name, 1)

        valid_freq = 50
        for i_epoch in range(exp_config.num_epochs):
            loss, _ = epoch_runner(train_batches, train_fn,
                                   valid_fn, valid_freq,
                                   save_fn, exp_config.save_interval)
        save_fn()



    def pred_ukp_aux(self, exp_config, topic):
        tf.reset_default_graph()
        task = transformer_nli(self.hparam, exp_config.voca_size, 0, True)
        data_loader = ukp.BertDataLoader(topic, True, self.hparam.seq_max, "bert_voca.txt", "only_topic_word")

        def load_available():
            run_dir = os.path.join(self.model_dir, 'runs')
            save_dir = os.path.join(run_dir, exp_config.name)
            model_id = None
            for (dirpath, dirnames, filenames) in os.walk(save_dir):
                for filename in filenames:
                    if ".meta" in filename:
                        print(filename)
                        model_id = filename[:-5]

            print(model_id)
            assert model_id is not None
            self.load_model_white2((exp_config.name, model_id), ['bert', "cls_dense"], False)

        def batch2feed_dict(batch):
            x0,x1,x2, y  = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y: y,
            }
            return feed_dict

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)
        load_available()

        dev_batches = get_batches_ex(data_loader.get_dev_data(), self.hparam.batch_size, 4)

        aux_keywords= data_loader.expand_topic(topic).split(",")[1:]

        def aux_batch(word):
            d = data_loader.get_dev_data_expand(word)
            return get_batches_ex(d, self.hparam.batch_size, 4)

        batches_aux = list([aux_batch(w) for w in aux_keywords])

        idx_for = data_loader.labels.index("Argument_for")
        idx_against = data_loader.labels.index("Argument_against")

        def forward(batches):
            logit_list = []
            gold_y = []

            for batch in batches:
                x0, x1, x2, y = batch
                loss_val, logits = self.sess.run([task.loss, task.logits],
                                                  feed_dict=batch2feed_dict(batch)
                                                  )
                logit_list.append(logits)
                gold_y.append(y)

            logit_list = np.concatenate(logit_list)
            gold_y = np.concatenate(gold_y)
            return logit_list, gold_y

        def print_result(logit_list, gold_y):
            pred_y = np.argmax(logit_list, axis=1)
            all_result = eval_3label(pred_y, gold_y)
            for_result = all_result[idx_for]
            against_result = all_result[idx_against]
            f1 = sum([result['f1'] for result in all_result]) / 3
            print("F1", f1)
            print("P_arg+", for_result['precision'])
            print("R_arg+", for_result['recall'])
            print("P_arg-", against_result['precision'])
            print("R_arg-", against_result['recall'])
            return all_result

        logit_raw, gold_y = forward(dev_batches)

        print("Raw run : ")
        print_result(logit_raw, gold_y)

        for word, batches in zip(aux_keywords, batches_aux):
            logit_aux1, gold_y = forward(batches)
            for weight in [0.1, 0.2, 0.4, 0.7]:
                logit_r = logit_raw - logit_aux1 * weight
                print("Keyword =  {} , weight={}".format(word, weight))
                print_result(logit_r, gold_y)


    def doc_stance(self, exp_config, data_loader):
        print("Train doc_stance")
        tf.reset_default_graph()
        task = transformer_distribution(self.hparam, exp_config.voca_size, True)
        with tf.variable_scope("optimizer"):
            train_cls = self.get_train_op(task.loss)

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(exp_config.name)
        train_batches = get_batches_ex(data_loader.get_train_data(), self.hparam.batch_size, 4)
        print("Num train batches :",len(train_batches))
        dev_batches = get_batches_ex(data_loader.get_dev_data(), self.hparam.batch_size, 4)
        self.load_model_white2(exp_config.preload_id, exp_config.load_names, False)

        def batch2feed_dict(batch):
            x0, x1, x2, y = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y: y,
            }
            return feed_dict

        def train_fn(batch, step_i):
            x0, x1, x2, y = batch
            #x0 = np.zeros_like(x0)
            #batch = x0, x1, x2, y
            _, pred, loss, \
                alpha, beta, gamma = self.sess.run([train_cls, task.pred, task.loss,
                                          task.alpha, task.beta, task.gamma,
                                             ],
                                            feed_dict=batch2feed_dict(batch)
                                            )
            self.log.debug("Step {0} train loss={1:.04f}".format(step_i, loss))

            for i in range(len(x0)):
                data_id = self.hparam.batch_size * step_i + i
                sents, label, topic = data_loader.train_docs[data_id]
                sent = " ".join(sents)
                sent = sent[:100].replace("\n", "\\n")
                p_str = " ".join(["{0:.2f}".format(pred[i,j]) for j in range(3)])
                g_str = " " .join(["{0:.2f}".format(y[i,j]) for j in range(3)])
                abc = [alpha[i], beta[i], gamma[i]]
                abc_str = " " .join(["{0:.2f}".format(v) for v in abc])
                self.log2.debug("ABC [{}]\t[{}]\t[{}]\t{}".format(abc_str, p_str, g_str, sent))
            self.g_step += 1
            return loss, 0

        def valid_fn():
            loss_list = []
            for batch in dev_batches:
                x0, x1, x2, y = batch
                pred, loss = self.sess.run([task.pred, task.loss
                                               ],
                                              feed_dict=batch2feed_dict(batch)
                                              )
                loss_list.append(loss)

            val_loss = average(loss_list)
            self.log.debug("Validation loss={0:.04f}".format(val_loss))
            return val_loss



        def save_fn():
            return self.save_model(exp_config.name, 1)

        valid_freq = 50
        print("Start training...")
        for i_epoch in range(exp_config.num_epochs):
            loss, _ = epoch_runner(train_batches, train_fn,
                                   valid_fn, valid_freq,
                                   save_fn, exp_config.save_interval, False)
        save_fn()

        val_loss = valid_fn()
        return val_loss


    def train_next_sent(self, exp_config, data_loader1, data_loader2, load_name):
        print("train_next_sent")

        tf.reset_default_graph()
        model = transformer_mt(self.hparam, exp_config.voca_size, [3, 3], True)
        with tf.variable_scope("optimizer"):
            train_cls1 = self.get_train_op(model.loss)
            train_cls2 = self.get_train_op(model.loss2)

        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge([model.summary['loss2'], model.summary['acc2']])
        self.setup_summary_writer(exp_config.name)
        print("Loading model : ", load_name)


        def load_available():
            run_dir = os.path.join(self.model_dir, 'runs')
            save_dir = os.path.join(run_dir, load_name)
            model_id = None
            for (dirpath, dirnames, filenames) in os.walk(save_dir):
                for filename in filenames:
                    if ".meta" in filename:
                        print(filename)
                        model_id = filename[:-5]

            print(model_id)
            assert model_id is not None
            self.load_model_white2((load_name, model_id), ['bert', "cls_dense"], False)

        load_available()
        #self.load_model_white2(preload_id, exp_config.load_names)

        def batch2feed_dict(y_idx, batch):
            x0,x1,x2, y  = batch
            feed_dict = {
                model.x_list[0]: x0,
                model.x_list[1]: x1,
                model.x_list[2]: x2,
                model.y[y_idx]: y,
            }
            return feed_dict

        def forward_run(inputs):
            batches = get_batches_ex(inputs, self.hparam.batch_size, 3)
            logit_list = []
            for batch in batches:
                x0, x1, x2 = batch
                logits,  = self.sess.run([model.sout, ],
                                               feed_dict={
                                                model.x_list[0]: x0,
                                                model.x_list[1]: x1,
                                                model.x_list[2]: x2,
                                               })
                logit_list.append(logits)
            return np.concatenate(logit_list)

        train_1_freq = 1
        def train_fn(batch, step_i):
            if step_i % train_1_freq == 0:
                task1_step = int(step_i / train_1_freq)
                train1(task1_step)
            return train2(batch, step_i)

        def train1(step_i):
            step_mod = step_i % len(train_batches1)
            batch = train_batches1[step_mod]
            loss_val, _ = self.sess.run([model.loss, train_cls1,
                                                  ],
                                                 feed_dict=batch2feed_dict(0, batch)
                                                 )
            self.log.debug("Step {0} train1 loss={1:.04f}".format(step_i, loss_val))
            return loss_val, 0

        def train2(batch, step_i):
            loss_val, summary, _ = self.sess.run([model.loss2, self.merged, train_cls2,
                                             ],
                                            feed_dict=batch2feed_dict(1, batch)
                                            )
            self.log.debug("Step {0} train2 loss={1:.04f}".format(step_i, loss_val))
            #self.train_writer.add_summary(summary, self.g_step)
            self.g_step += 1
            return loss_val, 0

        def get_train_data():
            return get_batches_ex(data_loader2.get_train_data(), self.hparam.batch_size, 4)
        print("Encoding train2")
        train_batches2 = self.pickle_cacher(exp_config.name + "train", get_train_data, True)
        print("Total of {} steps".format(len(train_batches2)))
        dev_batches1 = get_batches_ex(data_loader1.get_dev_data(), self.hparam.batch_size, 4)
        train_batches1 = get_batches_ex(data_loader1.get_train_data(), self.hparam.batch_size, 4)

        print("Encoding dev2")
        dev_batches2 = get_batches_ex(data_loader2.get_dev_data(), self.hparam.batch_size, 4)

        def get_f1(logit_list, gold_y_list):
            logit_list = np.concatenate(logit_list)
            gold_y = np.concatenate(gold_y_list)
            pred_y = np.argmax(logit_list, axis=1)

            all_result = eval_3label(pred_y, gold_y)
            f1 = sum([result['f1'] for result in all_result]) / 3
            return f1

        def valid_task2():
            loss_list = []
            logit_list = []
            gold_y = []

            for batch in dev_batches2:
                x0, x1, x2, y = batch
                loss_val, logits, summary = self.sess.run([model.loss2, model.logits2, self.merged],
                                                  feed_dict=batch2feed_dict(1, batch)
                                                  )
                loss_list.append(loss_val)
                logit_list.append(logits)
                gold_y.append(y)
                self.test_writer.add_summary(summary, self.g_step)

            self.log.info("Valid2 : NextPred loss={0:.04f}".format(average(loss_list)))
            self.log.info("Valid2 : NextPred F1={0:.04f}".format(get_f1(logit_list, gold_y)))


        def valid_stance():
            loss_list = []
            logit_list = []
            gold_y = []

            for batch in dev_batches1:
                x0, x1, x2, y = batch
                loss_val, logits = self.sess.run([model.loss, model.logits],
                                                  feed_dict=batch2feed_dict(0, batch)
                                                  )
                loss_list.append(loss_val)
                logit_list.append(logits)
                gold_y.append(y)
            self.log.info("Valid1 : Stance loss={0:.04f}".format(average(loss_list)))

            f1 = get_f1(logit_list, gold_y)
            #print("Stance F1", f1)
            self.log.info("Valid1 : Stance F1={0:.04f}".format(f1))



        valid2_freq = 10
        valid2_count = 0
        def valid_fn():
            nonlocal valid2_count
            if valid2_count % valid2_freq == 0:
                valid_stance()
            valid2_count += 1
            valid_task2()


        def save_fn():
            self.save_model(exp_config.name, 1)

        valid_freq = 10
        # train_op
        print("Train epoch")
        num_epochs = exp_config.num_epoch
        for i_epoch in range(num_epochs):
            loss, _ = epoch_runner(train_batches2, train_fn,
                                   valid_fn, valid_freq,
                                   save_fn, exp_config.save_interval)

    def train_crs_classify(self, e_config, data_loader, preload_id):
        print("Experiment.train_crs_classify()")
        valid_freq = 10
        task = CrsModel(self.hparam, e_config.voca_size, True)
        with tf.variable_scope("optimizer"):
            train_op = self.get_train_op(task.loss)

        print("loading model")
        self.sess = self.init_sess()
        self.sess.run(tf.global_variables_initializer())
        self.merged = tf.summary.merge_all()
        self.setup_summary_writer(e_config.name)
        self.load_model_white2(preload_id, e_config.load_names)

        random.seed(0)
        print("generating data")

        train_batches = get_batches_ex(data_loader.get_train_data(), self.hparam.batch_size, 7)
        dev_batches = get_batches_ex(data_loader.get_dev_data(), self.hparam.batch_size, 7)

        def batch2feed_dict(batch):
            x0,x1,x2, y0,y1, y0_sum, y1_sum  = batch
            feed_dict = {
                task.x_list[0]: x0,
                task.x_list[1]: x1,
                task.x_list[2]: x2,
                task.y[0]: y0,
                task.y[1]: y1,
                task.y_sum[0]: y0_sum,
                task.y_sum[1]: y1_sum,
            }
            return feed_dict


        def train_fn(batch, step_i):
            loss_val, summary, acc,  _ = self.sess.run([task.loss, self.merged, task.acc, train_op,
                                             ],
                                            feed_dict=batch2feed_dict(batch)
                                            )
            self.log.debug("Step {0} train loss={1:.04f} s_acc={2:.03f} d_acc={3:.03f}".format(step_i, loss_val, acc[0], acc[1]))
            self.train_writer.add_summary(summary, self.g_step)
            return loss_val, acc[0]


        valid_history = []

        def valid_fn():
            loss_list = []
            acc_list = []
            gold_list = []
            for batch in dev_batches:
                loss_val, summary, acc, = self.sess.run([task.loss, self.merged, task.acc,
                                                           ],
                                                          feed_dict=batch2feed_dict(batch)
                                                          )

                loss_list.append(loss_val)
                acc_list.append(acc)

            avg_loss = average(loss_list)
            s_acc_list, d_acc_list = zip(*acc_list)
            s_avg_acc = average(s_acc_list)
            d_avg_acc = average(d_acc_list)
            self.log.info("[Dev] S_Acc={0:.02f} D_Acc={1:.02f} loss={2:.04f}".format(s_avg_acc, d_avg_acc, avg_loss))
            return

        num_epoch = self.hparam.num_epochs
        print("Start Training")
        for i in range(num_epoch):
            loss, _ = epoch_runner(train_batches, train_fn,
                                   valid_fn, valid_freq,
                                   self.temp_saver, self.save_interval)
            self.log.info("[Train] Epoch {} Done. Loss={}".format(i, loss))

        valid_fn()
        self.save_model("crs")
