import os
import re

import tensorflow as tf
# from tensorflow_core.python.keras.api._v2 import keras
from tensorflow import keras

from cpath import data_path
from models.keras_model.bert_keras.modular_bert import BertClassifierLayer, define_bert_keras_inputs, BertClsProbe
from tlm.model.base import BertConfig


def load_stock_weights(model, ckpt_path, name_mapping, ignore_unused_prefixes=[]):
    assert len(model.weights) > 0, "BertModelLayer weights have not been instantiated yet. " \
                                  "Please add the layer in a Keras model and call model.build() first!"

    ckpt_reader = tf.train.load_checkpoint(ckpt_path)

    stock_weights = set(ckpt_reader.get_variable_to_dtype_map().keys())

    prefix = "bert"

    loaded_weights = set()
    skip_count = 0
    weight_value_tuples = []
    skipped_weight_value_tuples = []

    bert_params = model.weights
    param_values = keras.backend.batch_get_value(model.weights)
    for ndx, (param_value, param) in enumerate(zip(param_values, bert_params)):
        stock_name = name_mapping(param.name)
        # print("{} -> {}".format(param.name, stock_name))
        if stock_name and ckpt_reader.has_tensor(stock_name):
            ckpt_value = ckpt_reader.get_tensor(stock_name)

            if param_value.shape != ckpt_value.shape:
                print("loader: Skipping weight:[{}] as the weight shape:[{}] is not compatible "
                      "with the checkpoint:[{}] shape:{}".format(param.name, param.shape,
                                                                 stock_name, ckpt_value.shape))
                skipped_weight_value_tuples.append((param, ckpt_value))
                continue

            weight_value_tuples.append((param, ckpt_value))
            loaded_weights.add(stock_name)
        else:
            print("loader: No value for:[{}], i.e.:[{}] in:[{}]".format(param.name, stock_name, ckpt_path))
            skip_count += 1
    keras.backend.batch_set_value(weight_value_tuples)

    print("Done loading {} BERT weights from: {} into {} (prefix:{}). "
          "Count of weights not found in the checkpoint was: [{}]. "
          "Count of weights with mismatched shape: [{}]".format(
        len(weight_value_tuples), ckpt_path, model, prefix, skip_count, len(skipped_weight_value_tuples)))

    unused_weights = sorted(stock_weights.difference(loaded_weights))
    def skip(w_name):
        for prefix in ignore_unused_prefixes:
            if w_name.startswith(prefix):
                return True

        return False

    unused_weights_to_print = [w for w in unused_weights if not skip(w)]
    if unused_weights_to_print:
        print("Unused weights from checkpoint:",
              "\n\t" + "\n\t".join(unused_weights_to_print))

    return skipped_weight_value_tuples  # (bert_weight, value_from_ckpt)


def name_mapping_drop_colon(name):
    return name.split(":")[0]


def name_mapping_cls_probe(name):
    name = name.split(":")[0]
    name = re.sub(r'probe_layer(_\d*)?', 'match_predictor', name)
    return name


def load_model_from_v1_checkpoint(save_path, model_config):
    bert_config_file = os.path.join(data_path, "bert_config.json")
    bert_config = BertConfig.from_json_file(bert_config_file)
    bert_classifier_layer = BertClassifierLayer(bert_config, True, model_config.num_classes, False)

    max_seq_len = model_config.max_seq_length
    inputs = define_bert_keras_inputs(max_seq_len)
    cls_logits = bert_classifier_layer.call(inputs)
    load_stock_weights(bert_classifier_layer, save_path, name_mapping_drop_colon, ["optimizer"])

    output = cls_logits
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model, bert_classifier_layer


def load_model_cls_probe_from_v1_checkpoint(save_path, model_config):
    bert_config_file = os.path.join(data_path, "bert_config.json")
    bert_config = BertConfig.from_json_file(bert_config_file)
    bert_cls_probe = BertClsProbe(bert_config, True, model_config.num_classes, False)
    max_seq_len = model_config.max_seq_length
    inputs = define_bert_keras_inputs(max_seq_len)
    cls_logits = bert_cls_probe.call(inputs)
    load_stock_weights(bert_cls_probe, save_path, name_mapping_cls_probe, ["optimizer", "probe_optimizer"])

    output = cls_logits
    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model, bert_cls_probe
