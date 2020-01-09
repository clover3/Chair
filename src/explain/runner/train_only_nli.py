import logging
import sys

import tensorflow as tf

from cache import load_cache, save_to_pickle
from data_generator.NLI import nli
from data_generator.common import get_tokenizer
from data_generator.shared_setting import NLI
from log import log as log_module
from misc_lib import average
from models.transformer import hyperparams
from models.transformer.nli_base import transformer_nli_pooled
from trainer.model_saver import save_model, load_bert_v2, get_model_path, tf_logger
from trainer.tf_module import step_runner, get_nli_batches_from_data_loader
from trainer.tf_train_module import init_session, get_train_op2


def train_nli(hparam, nli_setting, run_name, num_steps, data, model_path):
    print("Train nil :", run_name)
    task = transformer_nli_pooled(hparam, nli_setting.vocab_size)
    train_cls = get_train_op2(task.loss, hparam.lr, 75000)

    train_batches, dev_batches = data
    print("train:", train_batches[0][0][0])
    print("dev:", dev_batches[0][0][0])

    log = log_module.train_logger()
    log.setLevel(logging.INFO)


    sess = init_session()
    sess.run(tf.global_variables_initializer())
    if model_path is not None:
        load_bert_v2(sess, model_path)

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
        return save_model(sess, run_name, global_step)

    print("{} train batches".format(len(train_batches)))
    valid_freq = 100
    save_interval = 1000
    loss, _ = step_runner(train_batches, train_classification,
                           valid_fn, valid_freq,
                           save_fn, save_interval, num_steps)

    return save_fn()


def train_nil_from_v2_checkpoint(run_name):
    hp = hyperparams.HPSENLI_grad_acc()
    print(hp.batch_size)
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"
    data_loader = nli.DataLoader(hp.seq_max, nli_setting.vocab_filename, True)
    tokenizer = get_tokenizer()
    CLS_ID = tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
    SEP_ID = tokenizer.convert_tokens_to_ids(["[SEP]"])[0]
    data_loader.CLS_ID = CLS_ID
    data_loader.SEP_ID = SEP_ID
    tf_logger.setLevel(logging.INFO)

    steps = 12271
    model_path = get_model_path("nli_batch16", 'model.ckpt-61358')
    data = load_cache("nli_batch16")
    if data is None:
        tf_logger.info("Encoding data from csv")
        data = get_nli_batches_from_data_loader(data_loader, hp.batch_size)
        save_to_pickle(data, "nli_batch16")
    train_nli(hp, nli_setting, run_name, steps, data, model_path)


if __name__  == "__main__":
    train_nil_from_v2_checkpoint(sys.argv[1])