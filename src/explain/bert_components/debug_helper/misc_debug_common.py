import numpy as np

from data_generator.NLI import nli
from trainer.np_modules import get_batches_ex


class ModelConfig:
    max_seq_length = 300
    num_classes = 3


def load_data(seq_max, batch_size):
    vocab_filename = "bert_voca.txt"
    data_loader = nli.DataLoader(seq_max, vocab_filename, True)
    dev_batches = get_batches_ex(data_loader.get_dev_data(), batch_size, 4)
    return dev_batches


def evaluate_acc_for_batches(batch_predict, dev_batches):
    y_all = []
    pred_all = []
    for batch in dev_batches:
        logits, y = batch_predict(batch)
        pred = np.argmax(logits, axis=1)
        y_all.append(y)
        pred_all.append(pred)
    pred_all = np.concatenate(pred_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    acc = np.average(pred_all == y_all)
    return acc