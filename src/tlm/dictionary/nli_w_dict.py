import logging

import numpy as np
import tensorflow as tf
from absl import app

import trainer.tf_train_module_v2 as train_module
from data_generator.NLI import nli
from log import log as log_module
from misc_lib import *
from path import output_path, get_bert_full_path, get_latest_model_path_from_dir_path
from tf_util.tf_logging import tf_logging
from tlm.dictionary.dict_augment import DictAugmentedDataLoader, NAME_DUMMY_WSSDR
from tlm.dictionary.dict_reader_interface import DictReaderInterface, DictReaderWrapper, WSSDRWrapper
from tlm.model_cnfig import JsonConfig
from tlm.training.dict_model_fn import get_bert_assignment_map_for_dict
from tlm.training.lm_model_fn import get_bert_assignment_map, get_assignment_map_as_is
from tlm.training.train_flags import *
from trainer.model_saver import save_model_to_dir_path, load_model, get_canonical_model_path
from trainer.tf_module import get_loss_from_batches


def setup_summary_writer(exp_name, sess):
    summary_path = os.path.join(output_path, "summary")
    exist_or_mkdir(summary_path)
    summary_run_path = os.path.join(summary_path, exp_name)
    exist_or_mkdir(summary_run_path)

    train_log_path = os.path.join(summary_run_path, "train")
    test_log_path = os.path.join(summary_run_path, "test")
    train_writer = tf.compat.v1.summary.FileWriter(train_log_path,
                                              sess.graph)
    test_writer = tf.compat.v1.summary.FileWriter(test_log_path,
                                             sess.graph)
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


def eval_nli_w_dict(run_name,
                    model: DictReaderInterface,
                    model_path,
                    data_feeder_loader):
    print("Eval nil :", run_name)
    tf_logging.debug("Building graph")
    batch_size = FLAGS.train_batch_size

    sess = train_module.init_session()
    sess.run(tf.compat.v1.global_variables_initializer())
    last_saved = get_latest_model_path_from_dir_path(model_path)
    dev_batches = data_feeder_loader.get_dev_feeder().get_all_batches(batch_size)

    tf_logging.info("Loading previous model from {}".format(last_saved))
    load_model(sess, last_saved)
    def valid_fn(step_i):
        loss_list = []
        acc_list = []
        for batch in dev_batches:
            loss_val, acc = sess.run([model.get_cls_loss(), model.get_acc()],
                                                   feed_dict=model.batch2feed_dict(batch)
                                                   )
            loss_list.append(loss_val)
            acc_list.append(acc)

        loss_val = average(loss_list)
        acc = average(acc_list)
        print("Dev loss={1:.04f} acc={2:.03f}".format(step_i, loss_val, acc))
        return average(acc_list)

    valid_fn(0)


class TrainConfig:
    def __init__(self, config_path):
        config = JsonConfig.from_json_file(config_path)
        self.learning_rate = config.learning_rate
        self.lookup_threshold = config.lookup_threshold
        self.lookup_min_step = config.lookup_min_step
        self.num_epochs = config.num_epochs



class NoLookupException(Exception):
    pass


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
        train_cls, global_step = train_module.get_train_op(model.get_cls_loss(), lr)
        train_lookup_op, global_step = train_module.get_train_op(model.get_lookup_loss(), lr2, global_step)

    sess = train_module.init_session()
    sess.run(tf.compat.v1.global_variables_initializer())

    train_writer, test_writer = setup_summary_writer(run_name, sess)

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

    dev_batches = []
    n_dev_batch = 100
    for _ in range(n_dev_batch):
        dev_batches.append(dev_data_feeder.get_random_batch(batch_size))

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
        term_ranks = np.argsort(logits[:, :, 1], axis=1)
        batch = train_data_feeder.augment_dict_info(data_indices, term_ranks)

        loss_val, acc, _ = sess.run([model.get_cls_loss(), model.get_acc(), train_cls],
                                    feed_dict=model.batch2feed_dict(batch)
                                    )
        log.info("Step {0} train loss={1:.04f} acc={2:.03f}".format(step_i, loss_val, acc))
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
        loss_list = []
        acc_list = []
        for batch in dev_batches:
            loss_val, acc = sess.run([model.get_cls_loss(), model.get_acc()],
                                                   feed_dict=model.batch2feed_dict(batch)
                                                   )
            loss_list.append(loss_val)
            acc_list.append(acc)

        loss_val = average(loss_list)
        acc = average(acc_list)
        log.info("Step {0} Dev loss={1:.04f} acc={2:.03f}".format(step_i, loss_val, acc))
        test_writer.add_summary(get_summary_obj(loss_val, acc), step_i)
        return average(acc_list)

    def save_fn():
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
        eval_nli_w_dict(run_name, model, model_path, augment_data_loader)



def main(_):
    dev_fn()

if __name__ == '__main__':
    flags.mark_flag_as_required("modeling")
    flags.mark_flag_as_required("init_checkpoint")
    flags.mark_flag_as_required("num_train_epochs")
    app.run(main)
