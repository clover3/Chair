import os
import sys

import tensorflow as tf

from cpath import data_path
from data_generator.NLI import nli
from models.keras_model.bert_keras.modular_bert import BertClassifierLayer, define_bert_keras_inputs
from models.keras_model.bert_keras.v1_load_util import load_stock_weights
from tlm.model.base import BertConfig
from trainer.np_modules import get_batches_ex


class ModelConfig:
    max_seq_length = 300
    num_classes = 3


def load_data(seq_max, batch_size):
    vocab_filename = "bert_voca.txt"
    data_loader = nli.DataLoader(seq_max, vocab_filename, True)
    dev_batches = get_batches_ex(data_loader.get_dev_data(), batch_size, 4)
    return dev_batches


def main():
    bert_config_file = os.path.join(data_path, "bert_config.json")
    bert_config = BertConfig.from_json_file(bert_config_file)
    model_config = ModelConfig()
    bert_classifier_layer = BertClassifierLayer(bert_config, True, model_config.num_classes, False)
    max_seq_len = model_config.max_seq_length
    inputs = define_bert_keras_inputs(max_seq_len)
    cls_logits = bert_classifier_layer.call(inputs)
    save_path = sys.argv[1]

    load_stock_weights(bert_classifier_layer, save_path, ["optimizer"])
    # last_layer_out = bert_classifier_layer.sequence_output[-1]
    # first_token_tensor = tf.squeeze(last_layer_out[:, 0:1, :], axis=1)

    model = tf.keras.Model(inputs=inputs, outputs=cls_logits)
    dev_batches = load_data(300, 2)
    x0, x1, x2, y = dev_batches[0]

    out_value = model.predict((x0, x1, x2))
    print(list(out_value[0]))


if __name__ == "__main__":
    main()
