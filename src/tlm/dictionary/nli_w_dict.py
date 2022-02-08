import logging

import numpy as np
import tensorflow as tf
from absl import app

import models.bert_util.bert_utils
import trainer.tf_train_module_v2 as train_module
from cache import save_to_pickle
from cpath import output_path, get_bert_full_path, get_latest_model_path_from_dir_path
from data_generator.NLI import nli
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import lmap
from log import log as log_module
from misc_lib import *
from tf_util.tf_logging import tf_logging
from tlm.dictionary.dict_augment import DictAugmentedDataLoader, NAME_DUMMY_WSSDR
from tlm.dictionary.dict_reader_interface import DictReaderInterface, DictReaderWrapper, WSSDRWrapper, APRWrapper
from tlm.model_cnfig import JsonConfig
from tlm.training.assignment_map import get_bert_assignment_map, get_assignment_map_as_is
from tlm.training.dict_model_fn import get_bert_assignment_map_for_dict
from tlm.training.train_flags import *
from trainer.model_saver import save_model_to_dir_path, load_model, get_canonical_model_path
from trainer.tf_module import get_loss_from_batches, split_tvars
from visualize.html_visual import Cell, HtmlVisualizer


def setup_summary_writer(exp_name):
    summary_path = os.path.join(output_path, "summary")
    exist_or_mkdir(summary_path)
    summary_run_path = os.path.join(summary_path, exp_name)
    exist_or_mkdir(summary_run_path)

    train_log_path = os.path.join(summary_run_path, "train")
    test_log_path = os.path.join(summary_run_path, "test")
    train_writer = tf.compat.v1.summary.FileWriter(train_log_path)
    test_writer = tf.compat.v1.summary.FileWriter(test_log_path)
    return train_writer, test_writer


def init_dict_model_with_bert(sess, init_checkpoint):
    tvars = tf.compat.v1.trainable_variables()
    map1, map2, init_vars = get_bert_assignment_map_for_dict(tvars, init_checkpoint)
    loader = tf.compat.v1.train.Saver(map1)
    loader.restore(sess, init_checkpoint)
    loader = tf.compat.v1.train.Saver(map2)
    loader.restore(sess, init_checkpoint)


def init_dict_model_with_nli_and_bert(sess, nli_checkpoint, bert_checkpoint):
    tvars = tf.compat.v1.trainable_variables()
    bert_to_nli, init_vars = get_bert_assignment_map(tvars, nli_checkpoint)
    loader = tf.compat.v1.train.Saver(bert_to_nli)
    loader.restore(sess, nli_checkpoint)

    _, bert_to_dict, init_vars = get_bert_assignment_map_for_dict(tvars, bert_checkpoint)
    loader = tf.compat.v1.train.Saver(bert_to_dict)
    loader.restore(sess, bert_checkpoint)


def debug_names(is_training):
    tf.compat.v1.disable_eager_execution()

    seq_max = 200
    lr = 1e-5
    batch_size = FLAGS.train_batch_size

    tf_logging.debug("Building graph")
    model = DictReaderWrapper(3, seq_max, is_training)

    with tf.compat.v1.variable_scope("optimizer"):
        train_cls, global_step = train_module.get_train_op(model.cls_loss, lr)
        train_lookup, global_step = train_module.get_train_op(model.lookup_loss, lr, global_step)

    sess = train_module.init_session()
    sess.run(tf.compat.v1.global_variables_initializer())
    tvars = tf.compat.v1.trainable_variables()

    for var in tvars:
        name = var.name
        print(name)


class WSSDRRunner:
    def __init__(self, model, lookup_augment_fn, sess=None):
        if sess is None:
            self.sess = train_module.init_session()
            self.sess.run(tf.compat.v1.global_variables_initializer())
        else:
            self.sess = sess

        self.model = model
        self.lookup_augment_fn = lookup_augment_fn

    def load_last_saved_model(self, model_path):
        last_saved = get_latest_model_path_from_dir_path(model_path)
        load_model(self.sess, last_saved)
        tf_logging.info("Loading previous model from {}".format(last_saved))

    def run_batches_wo_lookup(self, batches):
        return self._run_batches(batches, self.run_batch)

    def run_batches_w_lookup(self, batches):
        return self._run_batches(batches, self.run_batch_with_lookup)

    def _run_batches(self, batches, batch_run_fn):
        loss_list, acc_list = zip(*lmap(batch_run_fn, batches))
        loss_val = average(loss_list)
        acc = average(acc_list)
        return loss_val, acc

    def run_batch(self, batch):
        loss_val, acc = self.sess.run([self.model.get_cls_loss(), self.model.get_acc()],
                                      feed_dict=models.bert_util.bert_utils.batch2feed_dict_4_or_5_inputs(batch)
                                      )
        return loss_val, acc

    def get_term_rank(self, batch):
        logits, = self.sess.run([self.model.get_lookup_logits()],
                           feed_dict=models.bert_util.bert_utils.batch2feed_dict_4_or_5_inputs(batch)
                           )
        ranks = np.argsort(logits[:, :, 1], axis=1)
        return np.flip(ranks, axis=1)

    def run_batch_with_lookup(self, indice_n_batch):
        indice, batch = indice_n_batch
        term_ranks = self.get_term_rank(batch)
        batch = self.lookup_augment_fn(indice, term_ranks)
        return self.run_batch(batch)


def eval_nli_w_dict(run_name,
                    model: DictReaderInterface,
                    model_path,
                    data_feeder_loader):
    print("Eval nil :", run_name)
    tf_logging.debug("Building graph")
    batch_size = FLAGS.train_batch_size
    dev_data_feeder = data_feeder_loader.get_dev_feeder()
    dev_batches = dev_data_feeder.get_all_batches(batch_size)

    runner = WSSDRRunner(model, dev_data_feeder.augment_dict_info)
    runner.load_last_saved_model(model_path)

    def valid_fn(step_i):
        loss, acc = runner.run_batches_wo_lookup(dev_batches)
        print("Dev loss={1:.04f} acc={2:.03f}".format(step_i, loss, acc))
        return acc

    return valid_fn(0)

def eval_nli_w_dict_lookup(run_name,
                    model: DictReaderInterface,
                    model_path,
                    data_feeder_loader):
    print("eval_nli_w_dict_lookup :", run_name)
    tf_logging.debug("Building graph")
    batch_size = FLAGS.train_batch_size
    dev_data_feeder = data_feeder_loader.get_dev_feeder()
    runner = WSSDRRunner(model, dev_data_feeder.augment_dict_info)
    runner.load_last_saved_model(model_path)

    dev_batches = dev_data_feeder.get_all_batches(batch_size, True)[:100]
    n_batches = len(dev_batches)
    print('{} batches, about {} data'.format(n_batches, n_batches*batch_size))
    loss, acc = runner.run_batches_w_lookup(dev_batches)
    print("Dev total loss={0:.04f} acc={1:.03f}".format(loss, acc))
    return acc



def demo_nli_w_dict(run_name,
                    model: WSSDRWrapper,
                    model_path,
                    data_feeder_loader):
    print("Demonstrate nil_w_dict :", run_name)
    tf_logging.debug("Building graph")
    batch_size = FLAGS.train_batch_size

    dev_data_feeder = data_feeder_loader.get_dev_feeder()
    runner = WSSDRRunner(model, dev_data_feeder.augment_dict_info)
    runner.load_last_saved_model(model_path)
    dev_batches = dev_data_feeder.get_all_batches(batch_size, True)
    n_batches = len(dev_batches)
    tokenizer = get_tokenizer()
    html = HtmlVisualizer("nli_w_dict_demo.html")



    def fetch_fn(step_i):
        for indice, batch in dev_batches:
            print(indice)
            cache_name = "term_ranks_logits_cache"
            #logits = load_cache(cache_name)
            logits, = runner.sess.run([runner.model.get_lookup_logits()],
                                    feed_dict=models.bert_util.bert_utils.batch2feed_dict_4_or_5_inputs(batch)
                                    )
            raw_scores = logits[:, :, 1]
            term_ranks = np.argsort(logits[:, :, 1], axis=1)
            term_ranks = np.flip(term_ranks)
            save_to_pickle(logits, cache_name)

            x0, x1, x2, x3, y, x4, x5, x6, ab_map, ab_mapping_mask = batch

            for idx in range(len(indice)):
                ranks = term_ranks[idx]
                data_idx = indice[idx]
                input_ids = x0[idx]
                tokens = tokenizer.convert_ids_to_tokens(input_ids)

                words = dev_data_feeder.data_info[data_idx]
                location_to_word = dev_data_feeder.invert_index_word_locations(words)

                row = []
                for rank in term_ranks[idx]:
                    row.append(Cell(rank))

                for rank in ranks:
                    if rank in location_to_word and rank != 0:
                        highest_rank = rank
                        break
                for rank in ranks[::-1]:
                    if rank in location_to_word and rank != 0:
                        lowest_rank = rank
                        break


                html.write_table([row])
                t1 = []
                s1 = []
                t2 = []
                s2 = []
                text = [t1, t2]
                score_row = [s1, s2]

                sent_idx = 0
                for i, t in enumerate(tokens):
                    score = raw_scores[idx, i]
                    if i in location_to_word:
                        if i == highest_rank:
                            c = Cell(tokens[i], 150)
                        elif i == lowest_rank:
                            c = Cell(tokens[i], 150, target_color="R")
                        else:
                            c = Cell(tokens[i], 70)
                        s = Cell(score, score * 100)
                    else:
                        c = Cell(tokens[i])
                        s = Cell(score, score * 70)

                    text[sent_idx].append(c)
                    score_row[sent_idx].append(s)

                    if tokens[i] == "[unused3]":
                        sent_idx += 1
                        if sent_idx == 2:
                            break

                html.write_table([t1, s1])
                html.write_table([t2, s2])

                rows = []
                for word in words:
                    row = [Cell(word.word), Cell(word.location)]
                    rows.append(row)
                html.write_table(rows)




    fetch_fn(0)


class TrainConfig:
    def __init__(self, config_path):
        config = JsonConfig.from_json_file(config_path)
        self.learning_rate = config.learning_rate
        self.lookup_threshold = config.lookup_threshold
        self.lookup_min_step = config.lookup_min_step
        self.num_epochs = config.num_epochs



class NoLookupException(Exception):
    pass


def get_train_op_sep_lr(loss, lr, factor, scope_key, global_step = None, name='Adam'):
    if global_step is None:
        global_step = tf.Variable(0, name='global_step', trainable=False)
    all_vars = tf.compat.v1.trainable_variables()
    vars1, vars2 = split_tvars(all_vars, scope_key)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.98, epsilon=1e-8, name=name)
    train_op = optimizer.minimize(loss, global_step=global_step, var_list=vars1)
    optimizer2 = tf.compat.v1.train.AdamOptimizer(learning_rate=lr*factor, beta1=0.9, beta2=0.98, epsilon=1e-8, name=name)
    train_op2 = optimizer2.minimize(loss, global_step=global_step, var_list=vars2)

    train_op = tf.group([train_op, train_op2])
    return train_op, global_step


def train_nli_w_dict(run_name,
                     model: DictReaderInterface,
                     model_path,
                     model_config,
                     data_feeder_loader,
                     model_init_fn):
    print("Train nil :", run_name)
    batch_size = FLAGS.train_batch_size
    f_train_lookup = "lookup" in FLAGS.train_op
    tf_logging.debug("Building graph")

    with tf.compat.v1.variable_scope("optimizer"):
        lr = FLAGS.learning_rate
        lr2 = lr * 0.1
        if model_config.compare_attrib_value_safe("use_two_lr", True):
            tf_logging.info("Using two lr for each parts")
            train_cls, global_step = get_train_op_sep_lr(model.get_cls_loss(), lr, 5, "dict")
        else:
            train_cls, global_step = train_module.get_train_op(model.get_cls_loss(), lr)
        train_lookup_op, global_step = train_module.get_train_op(model.get_lookup_loss(), lr2, global_step)

    sess = train_module.init_session()
    sess.run(tf.compat.v1.global_variables_initializer())

    train_writer, test_writer = setup_summary_writer(run_name)

    last_saved = get_latest_model_path_from_dir_path(model_path)
    if last_saved:
        tf_logging.info("Loading previous model from {}".format(last_saved))
        load_model(sess, last_saved)
    elif model_init_fn is not None:
        model_init_fn(sess)

    log = log_module.train_logger()
    train_data_feeder = data_feeder_loader.get_train_feeder()
    dev_data_feeder = data_feeder_loader.get_dev_feeder()
    lookup_train_feeder = train_data_feeder
    valid_runner = WSSDRRunner(model, dev_data_feeder.augment_dict_info, sess)

    dev_batches = []
    n_dev_batch = 100
    dev_batches_w_dict = dev_data_feeder.get_all_batches(batch_size, True)[:n_dev_batch]
    for _ in range(n_dev_batch):
        dev_batches.append(dev_data_feeder.get_random_batch(batch_size))
        dev_batches_w_dict.append(dev_data_feeder.get_lookup_batch(batch_size))

    def get_summary_obj(loss, acc):
        summary = tf.compat.v1.Summary()
        summary.value.add(tag='loss', simple_value=loss)
        summary.value.add(tag='accuracy', simple_value=acc)
        return summary

    def get_summary_obj_lookup(loss, p_at_1):
        summary = tf.compat.v1.Summary()
        summary.value.add(tag='lookup_loss', simple_value=loss)
        summary.value.add(tag='P@1', simple_value=p_at_1)
        return summary

    def train_lookup(step_i):
        batches, info = lookup_train_feeder.get_lookup_train_batches(batch_size)
        if not batches:
            raise NoLookupException()

        def get_cls_loss(batch):
            return sess.run([model.get_cls_loss_arr()], feed_dict=model.batch2feed_dict(batch))

        loss_array = get_loss_from_batches(batches, get_cls_loss)

        supervision_for_lookup = train_data_feeder.get_lookup_training_batch(loss_array, batch_size, info)

        def lookup_train(batch):
            return sess.run([model.get_lookup_loss(), model.get_p_at_1(), train_lookup_op],
                            feed_dict=model.batch2feed_dict(batch))

        avg_loss, p_at_1, _ = lookup_train(supervision_for_lookup)
        train_writer.add_summary(get_summary_obj_lookup(avg_loss, p_at_1), step_i)
        log.info("Step {0} lookup loss={1:.04f}".format(step_i, avg_loss))
        return avg_loss

    def train_classification(step_i):
        batch = train_data_feeder.get_random_batch(batch_size)
        loss_val, acc, _ = sess.run([model.get_cls_loss(), model.get_acc(), train_cls],
                                                   feed_dict=model.batch2feed_dict(batch)
                                                   )
        log.info("Step {0} train loss={1:.04f} acc={2:.03f}".format(step_i, loss_val, acc))
        train_writer.add_summary(get_summary_obj(loss_val, acc), step_i)

        return loss_val, acc

    lookup_loss_window = MovingWindow(20)

    def train_classification_w_lookup(step_i):
        data_indices, batch = train_data_feeder.get_lookup_batch(batch_size)
        logits, = sess.run([model.get_lookup_logits()],
                                feed_dict=model.batch2feed_dict(batch)
                                )
        term_ranks = np.flip(np.argsort(logits[:, :, 1], axis=1))
        batch = train_data_feeder.augment_dict_info(data_indices, term_ranks)

        loss_val, acc, _ = sess.run([model.get_cls_loss(), model.get_acc(), train_cls],
                                    feed_dict=model.batch2feed_dict(batch)
                                    )
        log.info("ClsW]Step {0} train loss={1:.04f} acc={2:.03f}".format(step_i, loss_val, acc))
        train_writer.add_summary(get_summary_obj(loss_val, acc), step_i)

        return loss_val, acc

    def lookup_enabled(lookup_loss_window, step_i):
        return step_i > model_config.lookup_min_step\
               and lookup_loss_window.get_average() < model_config.lookup_threshold

    def train_fn(step_i):
        if lookup_enabled(lookup_loss_window, step_i):
            loss, acc = train_classification_w_lookup((step_i))
        else:
            loss, acc = train_classification(step_i)
        if f_train_lookup and step_i % model_config.lookup_train_frequency == 0:
            try:
                lookup_loss = train_lookup(step_i)
                lookup_loss_window.append(lookup_loss, 1)
            except NoLookupException:
                log.warning("No possible lookup found")

        return loss, acc

    def debug_fn(batch):
        y_lookup,  = sess.run([model.y_lookup, ],
                                                   feed_dict=model.batch2feed_dict(batch)
                                                   )
        print(y_lookup)
        return 0, 0

    def valid_fn(step_i):
        if lookup_enabled(lookup_loss_window, step_i):
            valid_fn_w_lookup(step_i)
        else:
            valid_fn_wo_lookup(step_i)

    def valid_fn_wo_lookup(step_i):
        loss_val, acc = valid_runner.run_batches_wo_lookup(dev_batches)
        log.info("Step {0} Dev loss={1:.04f} acc={2:.03f}".format(step_i, loss_val, acc))
        test_writer.add_summary(get_summary_obj(loss_val, acc), step_i)
        return acc

    def valid_fn_w_lookup(step_i):
        loss_val, acc = valid_runner.run_batches_w_lookup(dev_batches_w_dict)
        log.info("Step {0} DevW loss={1:.04f} acc={2:.03f}".format(step_i, loss_val, acc))
        test_writer.add_summary(get_summary_obj(loss_val, acc), step_i)
        return acc

    def save_fn():
        op = tf.compat.v1.assign(global_step, step_i)
        sess.run([op])
        return save_model_to_dir_path(sess, model_path, global_step)

    n_data = train_data_feeder.get_data_len()
    step_per_epoch = int((n_data+batch_size-1)/batch_size)
    tf_logging.debug("{} data point -> {} batches / epoch".format(n_data, step_per_epoch))
    train_steps = step_per_epoch * FLAGS.num_train_epochs
    tf_logging.debug("Max train step : {}".format(train_steps))
    valid_freq = 100
    save_interval = 60 * 20
    last_save = time.time()

    init_step, = sess.run([global_step])
    print("Initial step : ", init_step)
    for step_i in range(init_step, train_steps):
        if dev_fn is not None:
            if (step_i+1) % valid_freq == 0:
                valid_fn(step_i)

        if save_fn is not None:
            if time.time() - last_save > save_interval:
                save_fn()
                last_save = time.time()

        loss, acc = train_fn(step_i)

    return save_fn()


def init_from_bert(sess):
    tf_logging.info("Initializing model with bert ")
    init_dict_model_with_bert(sess, get_bert_full_path())


def nli_initializer(nli_checkpoint_path):
    def init_from_nli(sess):
        tf_logging.info("Initializing model with nli(main) and bert(dict reader)")
        init_dict_model_with_nli_and_bert(sess, nli_checkpoint_path, get_bert_full_path())
    return init_from_nli


def dict_reader_initializer(dict_reader_checkpoint):
    return as_is_initializer_wight_sanity_check(dict_reader_checkpoint,
                                                'cross_1_to_2', "Initializing model with nli(main) and bert(dict reader)")

def wssdr_initializer(dict_reader_checkpoint):
    return as_is_initializer_wight_sanity_check(dict_reader_checkpoint,
                                                'mr_key',
                                                "Initializing model with wssdr")

def as_is_initializer_wight_sanity_check(checkpoint, name_that_should_appear, init_description):
    def sanity_check(init_vars):
        key = name_that_should_appear
        matched = False
        for vname in init_vars:
            if key in vname:
                matched = True

        return matched

    def init_from_dict_reader(sess):
        tf_logging.info(init_description)
        tvars = tf.compat.v1.trainable_variables()
        map, init_vars = get_assignment_map_as_is(tvars, checkpoint)

        if not sanity_check(init_vars):
            for v in tvars:
                if v.name not in init_vars:
                    tf_logging.warn("Not initialized : ".format(v.name))
            raise KeyError

        loader = tf.compat.v1.train.Saver(map)
        loader.restore(sess, checkpoint)

    return init_from_dict_reader

def looks_like_path(path_like):
    return "/" in path_like or "\\" in path_like

def get_model_path(output_dir):
    if looks_like_path(output_dir):

        head, run_name = os.path.split(output_dir)
        if not run_name:
            _, run_name = os.path.split(head)
        model_path = output_dir
    else:
        assert "/" not in output_dir
        run_name = output_dir
        model_path = get_canonical_model_path(run_name)
    return model_path, run_name


def get_model(seq_max, modeling, is_training):
    if modeling == "dict_1":
        model = DictReaderWrapper(3, seq_max, is_training)
    elif modeling == "wssdr":
        ssdr_config = JsonConfig.from_json_file(FLAGS.model_config_file)
        model = WSSDRWrapper(3, ssdr_config, seq_max, is_training)
    elif modeling == "apr":
        ssdr_config = JsonConfig.from_json_file(FLAGS.model_config_file)
        model = APRWrapper(3, ssdr_config, seq_max, is_training)
    else:
        assert False

    return model

def get_checkpoint_init_fn():
    if FLAGS.is_bert_checkpoint:
        init_fn = init_from_bert
    elif FLAGS.checkpoint_type == "nli":
        init_fn = nli_initializer(FLAGS.init_checkpoint)
    elif FLAGS.checkpoint_type == "dict_reader":
        init_fn = dict_reader_initializer(FLAGS.init_checkpoint)
    elif FLAGS.checkpoint_type == "wssdr":
        init_fn = wssdr_initializer(FLAGS.init_checkpoint)
    else:
        raise KeyError("Checkpoint type is not specified")
    return init_fn


def dev_fn():
    tf.compat.v1.disable_eager_execution()

    if FLAGS.task_completion_mark:
        if os.path.exists(FLAGS.task_completion_mark):
            tf_logging.warn("Task already completed")

    tf_logging.setLevel(logging.INFO)
    if FLAGS.log_debug:
        tf_logging.setLevel(logging.DEBUG)
    seq_max = 200
    data_loader = nli.DataLoader(seq_max, "bert_voca.txt", True)

    is_training = FLAGS.do_train
    init_fn = get_checkpoint_init_fn()

    model_name = FLAGS.modeling
    if FLAGS.modeling == NAME_DUMMY_WSSDR:
        tf_logging.info("Using dummy WSSDR")
        model_name = "wssdr"

    model = get_model(seq_max, model_name, is_training)
    model_config = JsonConfig.from_json_file(FLAGS.model_config_file)

    # assert that the attribute exists. ( and it should be 0 or positive)
    assert model_config.lookup_threshold >= 0
    assert model_config.lookup_min_step >= 0
    assert model_config.lookup_train_frequency > 0
    augment_data_loader = DictAugmentedDataLoader(FLAGS.modeling, data_loader, FLAGS.use_cache)

    model_path, run_name = get_model_path(FLAGS.output_dir)

    if FLAGS.do_train:
        saved_model = train_nli_w_dict(run_name, model, model_path, model_config, augment_data_loader, init_fn)
        if FLAGS.task_completion_mark:
            f = open(FLAGS.task_completion_mark, "w")
            f.write("Done")
            f.close()

        tf.compat.v1.reset_default_graph()
        eval_nli_w_dict(run_name, model, saved_model, augment_data_loader)

    elif FLAGS.do_eval:
        eval_nli_w_dict_lookup(run_name, model, model_path, augment_data_loader)

    else:
        demo_nli_w_dict(run_name, model, model_path, augment_data_loader)

def main(_):
    dev_fn()

if __name__ == '__main__':
    flags.mark_flag_as_required("modeling")
    flags.mark_flag_as_required("init_checkpoint")
    flags.mark_flag_as_required("num_train_epochs")
    app.run(main)
