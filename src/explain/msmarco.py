import os

from cache import load_pickle_from
from cpath import data_path
from trainer.np_modules import get_batches_ex


class Hyperparam:
    '''Hyperparameters'''
    # data
    # training
    batch_size = 4  # alias = N
    lr = 2e-5  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    seq_max = 512 # Maximum number of words in a sentence. alias = T.
    # Feel free to increase this if you are ambitious.
    hidden_units = 768  # alias = C
    num_blocks = 12  # number of encoder/decoder blocks
    num_heads = 12
    dropout_rate = 0.1
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.
    intermediate_size = 3072
    type_vocab_size = 2
    num_classes = 2


class ExTrainConfig:
    vocab_filename = "bert_voca.txt"
    vocab_size = 30522
    seq_length = 512
    max_steps = 10000
    num_gpu = 1
    num_deletion = 20
    g_val = 0.5
    save_train_payload = False
    drop_thres = 0.3


def load_data(batch_size):
    train_data = load_pickle_from(os.path.join(data_path, "msmarco", "train.pickle"))
    dev_data = load_pickle_from(os.path.join(data_path, "msmarco", "dev.pickle"))

    train_batches = get_batches_ex(train_data, batch_size, 4)
    dev_batches = get_batches_ex(dev_data, batch_size, 4)
    return train_batches, dev_batches