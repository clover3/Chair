from models.classic.stopword import load_stopwords
from tensorflow import keras
import tensorflow as tf
from absl import app
from trainer import tf_module
from log import log as log_module
import logging

from models.transformer import hyperparams
from path import output_path, data_path, get_bert_full_path, get_latest_model_path
import trainer.tf_train_module_v2 as train_module
from tlm.training.model_fn import get_bert_assignment_map

from misc_lib import *
from data_generator.NLI import nli
from trainer.model_saver import save_model, load_bert_v2, load_model
from tlm.dictionary.model_fn import DictReaderModel
from tlm.model.base import BertConfig
from tlm.training.train_flags import *
from tlm.dictionary.data_gen import DictAugment
from tlm.tf_logging import tf_logging
from tlm.training.dict_model_fn import get_bert_assignment_map_for_dict

from data_generator import tokenizer_wo_tf as tokenization
from cache import load_from_pickle, save_to_pickle, load_cache

class DictReaderWrapper:
    def __init__(self, num_classes, seq_length, is_training):
        placeholder = tf.compat.v1.placeholder
        bert_config = BertConfig.from_json_file(os.path.join(data_path, "bert_config.json"))
        def_max_length = FLAGS.max_def_length
        loc_max_length = FLAGS.max_loc_length
        self.input_ids = placeholder(tf.int64, [None, seq_length])
        self.input_mask_ = placeholder(tf.int64, [None, seq_length])
        self.segment_ids = placeholder(tf.int64, [None, seq_length])

        self.d_input_ids = placeholder(tf.int64, [None, def_max_length])
        self.d_input_mask = placeholder(tf.int64, [None, def_max_length])
        self.d_location_ids = placeholder(tf.int64, [None, loc_max_length])

        self.y_cls = placeholder(tf.int64, [None])
        self.y_lookup = placeholder(tf.int64, [None, seq_length])

        self.network = DictReaderModel(
                config=bert_config,
                d_config=bert_config,
                is_training=is_training,
                input_ids=self.input_ids,
                input_mask=self.input_mask_,
                d_input_ids=self.d_input_ids,
                d_input_mask=self.d_input_mask,
                d_location_ids=self.d_location_ids,
                use_target_pos_emb=True,
                token_type_ids=self.segment_ids,
                use_one_hot_embeddings=False,
            )

        self.cls_logits = keras.layers.Dense(num_classes)(self.network.pooled_output)
        self.cls_loss_arr = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.cls_logits,
            labels=self.y_cls)
        self.cls_loss = tf.reduce_mean(self.cls_loss_arr)

        self.lookup_logits = keras.layers.Dense(2)(self.network.sequence_output)
        self.lookup_loss_arr = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.lookup_logits,
            labels=self.y_lookup)
        self.lookup_loss_per_example = tf.reduce_sum(self.lookup_loss_arr, axis=-1)
        self.lookup_loss = tf.reduce_mean(self.lookup_loss_per_example)
        self.acc = tf_module.accuracy(self.cls_logits, self.y_cls)

    def batch2feed_dict(self, batch):
        x0, x1, x2, x3, x4, x5, y= batch
        feed_dict = {
            self.input_ids: x0,
            self.input_mask_: x1,
            self.segment_ids: x2,
            self.d_input_ids: x3,
            self.d_input_mask: x4,
            self.d_location_ids: x5,
            self.y_cls: y,
        }
        return feed_dict


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


def train_nli_w_dict(run_name, num_epochs, data_loader, model_init_fn, dictionary, is_training):
    print("Train nil :", run_name)

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

    last_saved = get_latest_model_path(run_name)
    if last_saved:
        tf_logging.info("Loading previous model from {}".format(last_saved))
        load_model(sess, last_saved)
    elif model_init_fn is not None:
        model_init_fn(sess)

    log = log_module.train_logger()

    tf_logging.debug("Loading training data")
    train_data = data_loader.get_train_data()

    tf_logging.debug("Loading train_data_feeder")
    train_data_feeder = load_cache("train_data_feeder")
    if train_data_feeder is None:
        print("Parsing terms in training data")
        train_data_feeder = DictAugment(train_data, dictionary)
    save_to_pickle(train_data_feeder, "train_data_feeder")

    tf_logging.debug("Initializing dev batch")
    dev_data_feeder = load_cache("dev_data_feeder")
    if dev_data_feeder is None:
        dev_data_feeder = DictAugment(data_loader.get_dev_data(), dictionary)
    save_to_pickle(dev_data_feeder, "dev_data_feeder")

    dev_batches = []
    n_dev_batch = 100
    for _ in range(n_dev_batch):
        dev_batches.append(dev_data_feeder.get_random_batch(batch_size))

    def train_classification(step_i):
        batch = train_data_feeder.get_random_batch(batch_size)
        loss_val, acc,  _ = sess.run([model.cls_loss, model.acc, train_cls],
                                                   feed_dict=model.batch2feed_dict(batch)
                                                   )
        log.info("Step {0} train loss={1:.04f} acc={2:.03f}".format(step_i, loss_val, acc))
        return loss_val, acc

    def valid_fn(step_i):
        loss_list = []
        acc_list = []
        for batch in dev_batches:
            loss_val, acc = sess.run([model.cls_loss, model.acc],
                                                   feed_dict=model.batch2feed_dict(batch)
                                                   )
            loss_list.append(loss_val)
            acc_list.append(acc)
        log.info("Step {0} Dev loss={1:.04f} acc={2:.03f}".format(step_i, loss_val, acc))
        return average(acc_list)

    def save_fn():
        return save_model(sess, run_name, global_step)

    n_data = len(train_data)
    step_per_epoch = int((n_data+batch_size-1)/batch_size)
    tf_logging.debug("{} data point -> {} batches / epoch".format(n_data, step_per_epoch))
    train_steps = step_per_epoch * num_epochs
    tf_logging.debug("Max train step : {}".format(train_steps))
    valid_freq = 100
    save_interval = 60 * 20
    save_fn()
    last_save = time.time()
    for step_i in range(train_steps):
        if dev_fn is not None:
            if step_i % valid_freq == 0:
                valid_fn(step_i)

        if save_fn is not None:
            if time.time() - last_save > save_interval:
                save_fn()
                last_save = time.time()

        loss, acc = train_classification(step_i)

    return save_fn()


def init_from_bert(sess):
    tf_logging.info("Initializing model with bert ")
    init_dict_model_with_bert(sess, get_bert_full_path())


def nli_initializer(nli_checkpoint_path):
    def init_from_nli(sess):
        tf_logging.info("Initializing model with nli(main) and bert(dict reader)")
        init_dict_model_with_nli_and_bert(sess, nli_checkpoint_path, get_bert_full_path())
    return init_from_nli


def dev_fn():
    tf.compat.v1.disable_eager_execution()
    tf_logging.setLevel(logging.INFO)
    if FLAGS.log_debug:
        tf_logging.setLevel(logging.DEBUG)
    seq_max = 200
    data_loader = nli.DataLoader(seq_max, "bert_voca.txt", True)
    tf_logging.debug("Loading dictionary from pickle")
    d = load_from_pickle("webster")
    if FLAGS.is_bert_checkpoint:
        init_fn = init_from_bert
    else:
        init_fn = nli_initializer(FLAGS.init_checkpoint)

    saved_model = train_nli_w_dict("nli_first", 2, data_loader,
                                   init_fn, d, True)

def main(_):
    dev_fn()


if __name__ == '__main__':
    app.run(main)