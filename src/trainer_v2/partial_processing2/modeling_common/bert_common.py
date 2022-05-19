import math
import re

import bert
import tensorflow as tf
from bert import BertModelLayer, StockBertConfig
from bert.loader import map_to_stock_variable_name, _checkpoint_exists, bert_prefix, map_stock_config_to_params
from tensorflow import keras

from trainer_v2.chair_logging import c_log


class RunConfig:
    batch_size = 16
    num_classes = 3
    train_step = 49875
    eval_every_n_step = 100
    save_every_n_step = 5000
    learning_rate = 1e-5
    model_save_path = "saved_model"
    init_checkpoint = ""


class ModelConfig:
    max_seq_length = 300
    num_classes = 3


class BERT_CLS:
    def __init__(self, bert_params, config: ModelConfig):
        l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")
        max_seq_len = config.max_seq_length
        num_classes = config.num_classes

        l_input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
        l_token_type_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="segment_ids")
        seq_out = l_bert([l_input_ids, l_token_type_ids])  # [batch_size, max_seq_len, hidden_size]
        self.seq_out = seq_out
        first_token = seq_out[:, 0, :]
        pooler = tf.keras.layers.Dense(bert_params.hidden_size, activation=tf.nn.tanh, name="bert/pooler/dense")
        pooled = pooler(first_token)
        output = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)(pooled)
        model = keras.Model(inputs=[l_input_ids, l_token_type_ids], outputs=output, name="bert_model")
        self.model: keras.Model = model
        self.l_bert = l_bert
        self.pooler = pooler



@tf.function
def eval_fn(model, item, loss_fn, dev_loss, dev_acc):
    x1, x2, y = item
    prediction, _ = model([x1, x2], training=False)
    loss = loss_fn(y, prediction)
    dev_loss.update_state(loss)
    pred = tf.argmax(prediction, axis=1)
    dev_acc.update_state(y, pred)


def load_stock_weights(bert: BertModelLayer, ckpt_path,
                       map_to_stock_fn=map_to_stock_variable_name,
                       n_expected_restore=None,
                       ):
    assert isinstance(bert, BertModelLayer), "Expecting a BertModelLayer instance as first argument"
    assert _checkpoint_exists(ckpt_path), "Checkpoint does not exist: {}".format(ckpt_path)
    assert len(bert.weights) > 0, "BertModelLayer weights have not been instantiated yet. " \
                                  "Please add the layer in a Keras model and call model.build() first!"

    ckpt_reader = tf.train.load_checkpoint(ckpt_path)

    stock_weights = set(ckpt_reader.get_variable_to_dtype_map().keys())

    prefix = bert_prefix(bert)

    loaded_weights = set()
    skip_count = 0
    weight_value_tuples = []
    skipped_weight_value_tuples = []

    bert_params = bert.weights
    param_values = keras.backend.batch_get_value(bert.weights)
    for ndx, (param_value, param) in enumerate(zip(param_values, bert_params)):
        stock_name = map_to_stock_fn(param.name, prefix)

        if ckpt_reader.has_tensor(stock_name):
            ckpt_value = ckpt_reader.get_tensor(stock_name)

            if param_value.shape != ckpt_value.shape:
                c_log.warn("loader: Skipping weight:[{}] as the weight shape:[{}] is not compatible "
                      "with the checkpoint:[{}] shape:{}".format(param.name, param.shape,
                                                                 stock_name, ckpt_value.shape))
                skipped_weight_value_tuples.append((param, ckpt_value))
                continue

            weight_value_tuples.append((param, ckpt_value))
            loaded_weights.add(stock_name)
        else:
            ("loader: No value for:[{}], i.e.:[{}] in:[{}]".format(param.name, stock_name, ckpt_path))
            skip_count += 1
    keras.backend.batch_set_value(weight_value_tuples)
    if n_expected_restore is not None and n_expected_restore == len(weight_value_tuples):
        pass
    else:
        c_log.warn("Done loading {} BERT weights from: {} into {} (prefix:{}). "
              "Count of weights not found in the checkpoint was: [{}]. "
              "Count of weights with mismatched shape: [{}]".format(
            len(weight_value_tuples), ckpt_path, bert, prefix, skip_count, len(skipped_weight_value_tuples)))

        c_log.warn("Unused weights from checkpoint:",
              "\n\t" + "\n\t".join(sorted(stock_weights.difference(loaded_weights))))

    return skipped_weight_value_tuples  # (bert_weight, value_from_ckpt)


def load_pooler(pooler: tf.keras.layers.Dense, ckpt_path):
    """
    Use this method to load the weights from a pre-trained BERT checkpoint into a bert layer.

    :param pooler: a dense layer instance within a built keras model.
    :param ckpt_path: checkpoint path, i.e. `uncased_L-12_H-768_A-12/bert_model.ckpt` or `albert_base_zh/albert_model.ckpt`
    :return: list of weights with mismatched shapes. This can be used to extend
    the segment/token_type embeddings.
    """
    ckpt_reader = tf.train.load_checkpoint(ckpt_path)

    stock_weights = set(ckpt_reader.get_variable_to_dtype_map().keys())

    loaded_weights = set()
    skip_count = 0
    weight_value_tuples = []
    bert_params = pooler.weights
    for ndx, (param) in enumerate(bert_params):
        name = param.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            stock_name = m.group(1)
        else:
            stock_name = name

        if ckpt_reader.has_tensor(stock_name):
            ckpt_value = ckpt_reader.get_tensor(stock_name)
            weight_value_tuples.append((param, ckpt_value))
            loaded_weights.add(stock_name)
        else:
            print("{} not found".format(stock_name))
            skip_count += 1
    keras.backend.batch_set_value(weight_value_tuples)


def is_interesting_step(step_idx):
    interval = int(math.pow(10, int(math.log10(step_idx) - 1)))
    if step_idx < 100:
        return True
    elif step_idx % interval == 0:
        return True
    return False


def load_bert_config(bert_config_file):
    with tf.io.gfile.GFile(bert_config_file, "r") as reader:
        bc = StockBertConfig.from_json_string(reader.read())
        bert_params = map_stock_config_to_params(bc)
        is_brightmart_weights = bc["ln_type"] is not None
        bert_params.project_position_embeddings = not is_brightmart_weights  # ALBERT: False for brightmart/weights
        bert_params.project_embeddings_with_bias = not is_brightmart_weights  # ALBERT: False for brightmart/weights
    return bert_params


def load_bert_checkpoint(bert_cls, checkpoint_path):
    load_stock_weights(bert_cls.l_bert, checkpoint_path, n_expected_restore=197)
    load_pooler(bert_cls.pooler, checkpoint_path)