import os

import tensorflow as tf
from tensorflow import keras

from cpath import output_path, data_path
from data_generator.light_dataloader import LightDataLoader
from misc.show_checkpoint_vars import load_checkpoint_vars
from models.keras_model.bert_keras.modular_bert import define_bert_keras_inputs, BertLayer
from models.keras_model.bert_keras.v1_load_util import load_model_from_v1_checkpoint
from models.transformer.bert_common_v2 import create_initializer
from tlm.model.base import BertConfig
from trainer.np_modules import get_batches_ex


class ModelConfig:
    max_seq_length = 512
    num_classes = 1


def name_mapping(name):
    name = name.split(":")[0]
    name = name.replace("LayerNorm", "layer_normalization")
    name = name.replace("/embeddings", "")
    return name


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



def load_model_from_v1_checkpoint(save_path, model_config) -> tf.keras.Model:
    bert_config_file = os.path.join(data_path, "bert_config.json")
    bert_config = BertConfig.from_json_file(bert_config_file)
    with tf.compat.v1.variable_scope("bert"):
        bert_layer = BertLayer(bert_config, True, True)
        with tf.compat.v1.variable_scope("pooler") as name_scope:
            pooler = tf.keras.layers.Dense(bert_config.hidden_size,
                                                activation=tf.keras.activations.tanh,
                                                kernel_initializer=create_initializer(bert_config.initializer_range),
                                                name=name_scope.name + "/dense"
                                                )

    # bert_classifier_layer = BertClassifierLayer(bert_config, True, model_config.num_classes, False)

    max_seq_len = model_config.max_seq_length
    inputs = define_bert_keras_inputs(max_seq_len)
    sequence_output = bert_layer.call(inputs)
    last_layer = sequence_output[-1]
    first_token_tensor = tf.squeeze(last_layer[:, 0:1, :], axis=1)
    pooled = pooler(first_token_tensor)
    logits = tf.keras.layers.Dense(1, name="cls_dense")(pooled)
    model = tf.keras.Model(inputs=inputs, outputs=logits)
    load_stock_weights(model, save_path, name_mapping, ["optimizer"])
    return model


def get_predictor():
    model_config = ModelConfig()
    save_path = os.path.join(output_path, "model", "runs", "mmd_2M", "model.ckpt-200000")

    vars = load_checkpoint_vars(save_path)
    for name, val in vars.items():
        print(name, val.shape)

    model = load_model_from_v1_checkpoint(save_path, model_config)
    vocab_filename = "bert_voca.txt"
    voca_path = os.path.join(data_path, vocab_filename)
    data_loader = LightDataLoader(model_config.max_seq_length, voca_path)

    def predict(sent1, sent2):
        data = list(data_loader.from_pairs([(sent1, sent2)]))
        batch = get_batches_ex(data, 1, 4)[0]
        x0, x1, x2, y = batch
        logits = model((x0, x1, x2))
        return logits

    return predict


def main():
    predict = get_predictor()
    while True:
        sent1 = input("Query: ")
        sent2 = input("Passage: ")
        logits = predict(sent1, sent2)
        print((logits))
        print((sent1, sent2))


if __name__ == "__main__":
    main()
