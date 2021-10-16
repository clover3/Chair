import os
import sys

import tensorflow as tf

from cpath import output_path
from data_generator.NLI import nli
from explain.bert_components.misc_debug_common import evaluate_acc_for_batches
from models.keras_model.bert_keras.v1_load_util import load_model
from trainer.np_modules import get_batches_ex


class ModelConfig:
    max_seq_length = 300
    num_classes = 3


def load_data(seq_max, batch_size):
    vocab_filename = "bert_voca.txt"
    data_loader = nli.DataLoader(seq_max, vocab_filename, True)
    dev_batches = get_batches_ex(data_loader.get_dev_data(), batch_size, 4)
    return dev_batches


def eval_accuracy():
    save_path = sys.argv[1]
    model, model_config = load_model(save_path)

    dev_batches = load_data(model_config.max_seq_length, 80)

    ce = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE,
        from_logits=True
    )
    model.compile(loss=ce, metrics=['accuracy'])

    def batch_predict(batch):
        x0, x1, x2, y = batch
        # model.evaluate((x0, x1, x2), y)
        logits = model.predict((x0, x1, x2))
        return logits, y
    acc = evaluate_acc_for_batches(batch_predict, dev_batches)
    print("acc", acc)


def save_to_v2_model():
    save_path = sys.argv[1]
    model, model_config = load_model(save_path)

    save_path = os.path.join(output_path, "model", "runs", "standard_nli_v2_modular")
    model.save(save_path)


def load_v2_model():
    save_path = os.path.join(output_path, "model", "runs", "standard_nli_v2_modular")
    model = tf.keras.models.load_model(save_path)
    model.summary()



if __name__ == "__main__":
    load_v2_model()
