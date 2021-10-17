import os

import h5py
import numpy as np
import tensorflow as tf

from cpath import output_path, data_path
from data_generator.NLI import nli
from models.keras_model.bert_keras.bert_common_eager import get_shape_list_no_name, \
    create_attention_mask_from_input_mask, reshape_from_matrix
from models.keras_model.bert_keras.modular_bert import BertClassifierLayer, define_bert_keras_inputs
from models.keras_model.bert_keras.v1_load_util import load_model_from_v1_checkpoint
from models.transformer.bert import reshape_to_matrix
from tlm.model.base import BertConfig
from trainer.np_modules import get_batches_ex


class ModelConfig:
    max_seq_length = 300
    num_classes = 3


def load_model_from_weights(model_config):
    bert_config_file = os.path.join(data_path, "bert_config.json")
    bert_config = BertConfig.from_json_file(bert_config_file)
    bert_classifier_layer = BertClassifierLayer(bert_config, True, model_config.num_classes, False)
    max_seq_len = model_config.max_seq_length
    inputs = define_bert_keras_inputs(max_seq_len)
    cls_logits = bert_classifier_layer.call(inputs)
    output = cls_logits
    model = tf.keras.Model(inputs=inputs, outputs=output)
    save_path = os.path.join(output_path, "model", "runs", "standard_nli_v2_weights")
    model.load_weights(save_path, by_name=True)

    return bert_classifier_layer, model


def check_accuracy(model_config, model):

    vocab_filename = "bert_voca.txt"
    data_loader = nli.DataLoader(model_config.max_seq_length, vocab_filename, True)
    insts = get_batches_ex(data_loader.get_dev_data(), 32, 4)
    x0, x1, x2, y = insts[0]
    logits = model.predict((x0, x1, x2))
    pred = np.argmax(logits, axis=1)
    acc = np.average(pred == y)
    if acc < 0.75:
        raise ValueError("Accuracy is low: {}".format(acc))


def load_h5_file():
    save_path = os.path.join(output_path, "model", "runs", "standard_nli_v2_weights")
    file = h5py.File(save_path, 'r')
    return file


def load_common_300_model():
    model_config = ModelConfig()
    save_path = os.path.join(output_path, "model", "runs", "standard_nli", "model-73630")
    model, bert_classifier_layer = load_model_from_v1_checkpoint(save_path, model_config)
    vocab_filename = "bert_voca.txt"
    data_loader = nli.DataLoader(model_config.max_seq_length, vocab_filename, True)
    dev_insts = data_loader.get_dev_data()
    return bert_classifier_layer, dev_insts


def load_cls_probe():
    model_config = ModelConfig()
    save_path = os.path.join(output_path, "model", "runs", "nli_probe_cls3", "model-100000")
    model, bert_classifier_layer = load_model_from_v1_checkpoint(save_path, model_config)
    vocab_filename = "bert_voca.txt"
    data_loader = nli.DataLoader(model_config.max_seq_length, vocab_filename, True)
    dev_insts = data_loader.get_dev_data()
    return bert_classifier_layer, dev_insts


def partial_execution(bert_cls: BertClassifierLayer,
                      X, Y):
    input_ids, input_mask, segment_ids = X
    raw_embedding = bert_cls.bert_layer.embedding_layer((input_ids, segment_ids))
    attention_mask = create_attention_mask_from_input_mask(
        input_ids, input_mask)

    embedding = bert_cls.bert_layer.embedding_layer_norm(raw_embedding)
    input_shape = get_shape_list_no_name(embedding)
    batch_size = input_shape[0]
    seq_length = input_shape[1]

    prev_output = reshape_to_matrix(embedding)
    shape_info = (batch_size, seq_length, attention_mask)
    for i in range(11):
        prev_output = bert_cls.bert_layer.layers[i]((prev_output, shape_info))

    prev_output = bert_cls.bert_layer.layers[11]((prev_output, shape_info))
    last_layer = reshape_from_matrix(prev_output, input_shape)
    first_token_tensor = tf.squeeze(last_layer[:, 0:1, :], axis=1)
    pooled = bert_cls.pooler(first_token_tensor)
    logits = bert_cls.named_linear(pooled)
    acc = logits_to_accuracy(logits, Y)
    print("accuracy", acc)


def logits_to_accuracy(logits, y):
    pred = np.argmax(logits, axis=1)
    acc = np.average(pred == y)
    return acc


def main():
    bert_classifier_layer, dev_insts = load_cls_probe()
    batches = get_batches_ex(dev_insts, 32, 4)
    x0, x1, x2, y = batches[0]
    X = (x0, x1, x2)
    logits = bert_classifier_layer(X)
    acc = logits_to_accuracy(logits, y)
    print("logits from eager", logits)
    print("accuracy", acc)
    if acc < 0.75:
        raise Exception()
    # partial_execution(bert_classifier_layer, X, y)


if __name__ == "__main__":
    main()


