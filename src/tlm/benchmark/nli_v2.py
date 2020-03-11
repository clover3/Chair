import tensorflow as tf

import data_generator.NLI.nli_info
from data_generator.NLI import nli
from data_generator.shared_setting import NLI
from log import log as log_module
from misc_lib import *
from models.transformer.hyperparams import HPBert
from tf_v2_support import placeholder, variable_scope
from tlm.benchmark.nli import save_report
from tlm.dictionary.nli_w_dict import setup_summary_writer
from tlm.model.base import BertModel, BertConfig
from tlm.training.assignment_map import get_bert_assignment_map
from trainer import tf_module
from trainer.model_saver import save_model
from trainer.tf_module import epoch_runner, get_nli_batches_from_data_loader
from trainer.tf_train_module_v2 import init_session, get_train_op


class Classification:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def predict(self, enc, Y, is_train):
        if is_train:
            mode = tf.estimator.ModeKeys.TRAIN
        else:
            mode = tf.estimator.ModeKeys.EVAL
        return self.predict_ex(enc, Y, mode)

    def predict_ex(self, enc, Y, mode):
        feature_loc = 0
        pooled = enc[:,feature_loc,:]
        logits = tf.keras.layers.Dense(self.num_classes, name="cls_dense")(pooled)
        preds = tf.argmax(logits, axis=-1)
        self.acc = tf_module.accuracy(logits, Y)
        self.logits = logits
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
            self.loss_arr = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=Y)
            self.loss = tf.reduce_mean(self.loss_arr)
            return preds, self.loss
        else:
            return preds


class transformer_nli:
    def __init__(self, hp, voca_size, method, is_training=True):
        config = BertConfig(vocab_size=voca_size,
                             hidden_size=hp.hidden_units,
                             num_hidden_layers=hp.num_blocks,
                             num_attention_heads=hp.num_heads,
                             intermediate_size=hp.intermediate_size,
                             type_vocab_size=hp.type_vocab_size,
                             )

        seq_length = hp.seq_max
        use_tpu = False
        task = Classification(data_generator.NLI.nli_info.num_classes)

        input_ids = placeholder(tf.int64, [None, seq_length])
        input_mask = placeholder(tf.int64, [None, seq_length])
        segment_ids = placeholder(tf.int64, [None, seq_length])
        label_ids = placeholder(tf.int64, [None])

        self.x_list = [input_ids, input_mask, segment_ids]
        self.y = label_ids

        use_one_hot_embeddings = use_tpu
        self.model = BertModel(
            config=config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        pred, loss = task.predict(self.model.get_sequence_output(), label_ids, True)

        self.logits = task.logits
        self.sout = tf.nn.softmax(self.logits)
        self.pred = pred
        self.loss = loss
        self.acc = task.acc


def init_model_with_bert(sess, init_checkpoint):
    tvars = tf.compat.v1.trainable_variables()
    map1, init_vars = get_bert_assignment_map(tvars, init_checkpoint)
    loader = tf.compat.v1.train.Saver(map1)
    loader.restore(sess, init_checkpoint)


def train_nli(hparam, nli_setting, run_name, num_epochs, data, model_path):
    print("Train nil :", run_name)
    task = transformer_nli(hparam, nli_setting.vocab_size, 2)
    with variable_scope("optimizer"):
        train_cls, global_step = get_train_op(task.loss, hparam.lr)

    train_batches, dev_batches = data

    log = log_module.train_logger()

    def get_summary_obj(loss, acc):
        summary = tf.compat.v1.Summary()
        summary.value.add(tag='loss', simple_value=loss)
        summary.value.add(tag='accuracy', simple_value=acc)
        return summary

    sess = init_session()
    sess.run(tf.compat.v1.global_variables_initializer())
    train_writer, test_writer = setup_summary_writer(run_name)
    if model_path is not None:
        init_model_with_bert(sess, model_path)

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
        loss_val, acc,  _ = sess.run([task.loss, task.acc, train_cls],
                                       feed_dict=batch2feed_dict(batch)
                                       )
        log.debug("Step {0} train loss={1:.04f} acc={2:.03f}".format(step_i, loss_val, acc))
        train_writer.add_summary(get_summary_obj(loss_val, acc), step_i)
        nonlocal g_step_i
        g_step_i = step_i
        return loss_val, acc


    def valid_fn():
        loss_list = []
        acc_list = []
        for batch in dev_batches[:100]:
            loss_val, acc = sess.run([task.loss, task.acc],
                                                   feed_dict=batch2feed_dict(batch)
                                                   )
            loss_list.append(loss_val)
            acc_list.append(acc)

        loss_val = average(loss_list)
        acc = average(acc_list)
        test_writer.add_summary(get_summary_obj(loss_val, acc), g_step_i)

        return average(acc_list)

    def save_fn():
        return save_model(sess, run_name, global_step)

    print("{} train batches".format(len(train_batches)))
    valid_freq = 100
    save_interval = 100000
    for i_epoch in range(num_epochs):
        loss, _ = epoch_runner(train_batches, train_classification,
                               valid_fn, valid_freq,
                               save_fn, save_interval)

    return save_fn()


def test_nli(hparam, nli_setting, run_name, data, model_path):
    print("test nil :", run_name)
    task = transformer_nli(hparam, nli_setting.vocab_size, 2, False)

    train_batches, dev_batches = data


    sess = init_session()
    sess.run(tf.global_variables_initializer())

    loader = tf.train.Saver(tf.global_variables(), max_to_keep=1)
    loader.restore(sess, model_path)

    def batch2feed_dict(batch):
        x0, x1, x2, y  = batch
        feed_dict = {
            task.x_list[0]: x0,
            task.x_list[1]: x1,
            task.x_list[2]: x2,
            task.y: y,
        }
        return feed_dict

    def valid_fn():
        loss_list = []
        acc_list = []
        for batch in dev_batches[:100]:
            loss_val, acc = sess.run([task.loss, task.acc],
                                                   feed_dict=batch2feed_dict(batch)
                                                   )
            loss_list.append(loss_val)
            acc_list.append(acc)

        return average(acc_list)

    return valid_fn()



def run_nli_w_path(run_name, step_name, model_path):
    #run_name
    hp = HPBert()
    nli_setting = NLI()
    nli_setting.vocab_size = 30522
    nli_setting.vocab_filename = "bert_voca.txt"

    data_loader = nli.DataLoader(hp.seq_max, "bert_voca.txt", True)
    data = get_nli_batches_from_data_loader(data_loader, hp.batch_size)
    run_name = "{}_{}_NLI".format(run_name, step_name)
    saved_model = train_nli(hp, nli_setting, run_name, 3, data, model_path)
    tf.reset_default_graph()
    avg_acc = test_nli(hp, nli_setting, run_name, data, saved_model)
    print("avg_acc: ", avg_acc)

    save_report("nli", run_name, step_name, avg_acc)

