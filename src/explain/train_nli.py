
import logging

import tensorflow as tf

from cache import load_cache, save_to_pickle
from data_generator.NLI import nli
from data_generator.common import get_tokenizer
from data_generator.shared_setting import BertNLI
from log import log as log_module
from misc_lib import average
from models.transformer import hyperparams
from models.transformer.nli_base import transformer_nli_pooled
from trainer.model_saver import save_model_to_dir_path, load_bert_v2, tf_logger
from trainer.tf_module import step_runner, get_nli_batches_from_data_loader
from trainer.tf_train_module import get_train_op2, init_session, get_train_op_from_grads_and_tvars


def train_nli(hparam, nli_setting, save_dir, num_steps, data, model_path, load_fn):
    print("Train nil :", save_dir)

    task = transformer_nli_pooled(hparam, nli_setting.vocab_size)
    with tf.variable_scope("optimizer"):
        train_cls = get_train_op2(task.loss, hparam.lr, "adam", num_steps)

    train_batches, dev_batches = data

    log = log_module.train_logger()
    log.setLevel(logging.INFO)

    sess = init_session()
    sess.run(tf.global_variables_initializer())
    if model_path is not None:
        load_fn(sess, model_path)

    def batch2feed_dict(batch):
        x0, x1, x2, y  = batch
        feed_dict = {
            task.x_list[0]: x0,
            task.x_list[1]: x1,
            task.x_list[2]: x2,
            task.y: y,
        }
        return feed_dict

    g_step_i = 0
    def train_classification(batch, step_i):
        loss_val, acc,  _ = sess.run([task.loss, task.acc, train_cls,
                                                    ],
                                                   feed_dict=batch2feed_dict(batch)
                                                   )
        log.debug("Step {0} train loss={1:.04f} acc={2:.03f}".format(step_i, loss_val, acc))
        g_step_i = step_i
        return loss_val, acc

    global_step = tf.train.get_or_create_global_step()

    def valid_fn():
        loss_list = []
        acc_list = []
        for batch in dev_batches:
            loss_val, acc, g_step_val = sess.run([task.loss, task.acc, global_step],
                                                   feed_dict=batch2feed_dict(batch)
                                                   )
            loss_list.append(loss_val)
            acc_list.append(acc)
        log.info("Step dev step={0} loss={1:.04f} acc={2:.03f}".format(g_step_val, average(loss_list), average(acc_list)))

        return average(acc_list)

    def save_fn():
        return save_model_to_dir_path(sess, save_dir, global_step)

    print("{} train batches".format(len(train_batches)))
    valid_freq = 5000
    save_interval = 10000
    loss, _ = step_runner(train_batches, train_classification,
                           valid_fn, valid_freq,
                           save_fn, save_interval, num_steps)

    return save_fn()


def get_multiple_models(model_init_fn, n_gpu):
    models = []
    root_scope = tf.get_variable_scope()


    for gpu_idx in range(n_gpu):
        print(gpu_idx)
        with tf.device("/gpu:{}".format(gpu_idx)):
            with tf.variable_scope(root_scope, reuse= gpu_idx >0):
                models.append(model_init_fn())

    return models


def get_averaged_gradients(models):
    tvars = tf.trainable_variables()
    n_gpu = len(models)

    tower_grads = []
    for gpu_idx, model in enumerate(models):
        with tf.device("/gpu:{}".format(gpu_idx)):
            grads = tf.gradients(get_loss(model), tvars)
            tower_grads.append(grads)

    avg_grads = []
    for t_idx, _ in enumerate(tvars):
        first_item = tower_grads[0][t_idx]
        if first_item is not None:
            try:
                g_list = [tower_grads[gpu_idx][t_idx] for gpu_idx in range(n_gpu)]
                g_avg = tf.reduce_sum(g_list) / n_gpu
            except TypeError:
                g_list = [t.values for t in g_list]
                g_avg = tf.reduce_sum(g_list) / n_gpu
        else:
            g_avg = None

        avg_grads.append(g_avg)
    return avg_grads

def get_loss(model):
    return model.loss


def get_avg_loss(models):
    return get_avg_tensors_from_model(models, get_loss)


def get_avg_tensors_from_model(models, get_tensor_fn):
    sum = 0
    for model in models:
        sum += get_tensor_fn(model)
    return sum / len(models)


def train_nli_multi_gpu(hparam, nli_setting, save_dir, num_steps, data, model_path, load_fn, n_gpu):
    print("Train nil :", save_dir)
    model_init_fn = lambda :transformer_nli_pooled(hparam, nli_setting.vocab_size)
    models = get_multiple_models(model_init_fn, n_gpu)
    gradients = get_averaged_gradients(models)
    avg_loss = get_avg_loss(models)
    avg_acc = get_avg_tensors_from_model(models, lambda model:model.acc)

    with tf.variable_scope("optimizer"):
        with tf.device("/device:CPU:0"):
            train_cls = get_train_op_from_grads_and_tvars(gradients, tf.trainable_variables(), hparam.lr, "adam", num_steps)

    train_batches, dev_batches = data

    log = log_module.train_logger()
    log.setLevel(logging.DEBUG)

    sess = init_session(True, True)
    sess.run(tf.global_variables_initializer())
    if model_path is not None:
        load_fn(sess, model_path)

    def batch2feed_dict(batch):
        x0, x1, x2, y  = batch
        batch_size = len(x0)
        batch_size_per_gpu = int(batch_size / n_gpu)

        feed_dict = {}
        for gpu_idx in range(n_gpu):
            st = batch_size_per_gpu * gpu_idx
            ed = batch_size_per_gpu * (gpu_idx + 1)
            local_feed_dict = {
                models[gpu_idx].x_list[0]: x0[st:ed],
                models[gpu_idx].x_list[1]: x1[st:ed],
                models[gpu_idx].x_list[2]: x2[st:ed],
                models[gpu_idx].y: y[st:ed],
            }
            feed_dict.update(local_feed_dict)
        return feed_dict

    g_step_i = 0
    def train_classification(batch, step_i):
        loss_val, acc, _ = sess.run([avg_loss, avg_acc, train_cls,
                                                    ],
                                                   feed_dict=batch2feed_dict(batch)
                                                   )
        log.debug("Step {0} train loss={1:.04f} acc={2:.04f}".format(step_i, loss_val, acc))
        g_step_i = step_i
        return loss_val, 0

    global_step = tf.train.get_or_create_global_step()

    def valid_fn():
        loss_list = []
        acc_list = []
        for batch in dev_batches[:10]:
            loss_val, acc, g_step_val = sess.run([avg_loss, avg_acc, global_step],
                                                   feed_dict=batch2feed_dict(batch)
                                                   )
            loss_list.append(loss_val)
            acc_list.append(acc)
        log.info("Step dev step={0} loss={1:.04f} acc={2:.03f}".format(g_step_val, average(loss_list), average(acc_list)))

        return average(acc_list)

    def save_fn():
        return save_model_to_dir_path(sess, save_dir, global_step)

    print("{} train batches".format(len(train_batches)))
    valid_freq = 5000
    save_interval = 10000
    loss, _ = step_runner(train_batches, train_classification,
                           valid_fn, valid_freq,
                           save_fn, save_interval, num_steps)

    return save_fn()


def eval_nli(hparam, nli_setting, save_dir, data, model_path, load_fn):
    print("Train nil :", save_dir)

    task = transformer_nli_pooled(hparam, nli_setting.vocab_size)

    train_batches, dev_batches = data

    log = log_module.train_logger()
    log.setLevel(logging.INFO)

    sess = init_session()
    sess.run(tf.global_variables_initializer())
    load_fn(sess, model_path)

    def batch2feed_dict(batch):
        x0, x1, x2, y  = batch
        feed_dict = {
            task.x_list[0]: x0,
            task.x_list[1]: x1,
            task.x_list[2]: x2,
            task.y: y,
        }
        return feed_dict

    global_step = tf.train.get_or_create_global_step()

    def valid_fn():
        return 0
        loss_list = []
        acc_list = []
        for batch in dev_batches:
            loss_val, acc, g_step_val = sess.run([task.loss, task.acc, global_step],
                                                   feed_dict=batch2feed_dict(batch)
                                                   )
            loss_list.append(loss_val)
            acc_list.append(acc)
        log.info("Step dev step={0} loss={1:.04f} acc={2:.03f}".format(g_step_val, average(loss_list), average(acc_list)))

        return average(acc_list)
    return valid_fn()

def train_nil_from_v2_checkpoint(run_name, model_path):
    steps = 12271
    return train_nil_from(run_name, model_path, load_bert_v2, steps)


def train_nil_from(save_dir, model_path, load_fn, steps):
    hp = hyperparams.HPSENLI3()
    tf_logger.setLevel(logging.INFO)
    nli_setting = BertNLI()
    data = get_nli_data(hp, nli_setting)
    train_nli(hp, nli_setting, save_dir, steps, data, model_path, load_fn)


def get_nli_data(hp, nli_setting):
    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    tokenizer = get_tokenizer()
    CLS_ID = tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
    SEP_ID = tokenizer.convert_tokens_to_ids(["[SEP]"])[0]
    data_loader.CLS_ID = CLS_ID
    data_loader.SEP_ID = SEP_ID
    cache_name = "nli_batch{}_seq{}".format(hp.batch_size, hp.seq_max)
    data = load_cache(cache_name)
    if data is None:
        tf_logger.info("Encoding data from csv")
        data = get_nli_batches_from_data_loader(data_loader, hp.batch_size)
        save_to_pickle(data, cache_name)
    return data