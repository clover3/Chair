import os

import tensorflow as tf

from cpath import output_path, data_path
from data_generator.light_dataloader import LightDataLoader
from explain.bert_components.nli300 import ModelConfig
from models.keras_model.bert_keras.v1_load_util import load_model_from_v1_checkpoint
from trainer.np_modules import get_batches_ex


def get_nli_predictor():
    model_config = ModelConfig()
    save_path = os.path.join(output_path, "model", "runs", "standard_nli", "model-73630")
    model, bert_classifier_layer = load_model_from_v1_checkpoint(save_path, model_config)
    vocab_filename = "bert_voca.txt"
    voca_path = os.path.join(data_path, vocab_filename)
    data_loader = LightDataLoader(model_config.max_seq_length, voca_path)

    def predict(sent1, sent2):
        data = list(data_loader.from_pairs([(sent1, sent2)]))
        batch = get_batches_ex(data, 1, 4)[0]
        x0, x1, x2, y = batch
        logits = bert_classifier_layer((x0, x1, x2))
        probs = tf.nn.softmax(logits)
        return probs[0]

    return predict