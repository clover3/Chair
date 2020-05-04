import os

from cache import load_pickle_from
from cpath import data_path
from explain.ex_train_modules import action_penalty
from explain.runner.train_ex import train_self_explain
from explain.setups import init_fn_generic
from tf_util.tf_logging import tf_logging
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

def tag_informative(tag, before_prob, after_prob, action):
    score = before_prob[1] - after_prob[1]
    score = score - action_penalty(action)
    return score


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


def get_params(start_model_path, start_type, num_gpu):
    hp = Hyperparam()
    data = load_data(hp.batch_size)
    # Data : Tuple[train batches, dev batches]
    data_loader = None
    train_config = ExTrainConfig()
    train_config.num_gpu = num_gpu

    def init_fn(sess):
        return init_fn_generic(sess, start_type, start_model_path)

    return data, data_loader, hp, tag_informative, init_fn, train_config


def train_from(start_model_path,
               start_type,
               save_dir,
               modeling_option,
               num_deletion,
               g_val=0.5,
               num_gpu=1,
               drop_thres=0.3,):

    num_deletion = int(num_deletion)
    num_gpu = int(num_gpu)
    tf_logging.info("train_from : msmarco_ex")
    data, data_loader, hp, informative_fn, init_fn, train_config\
        = get_params(start_model_path, start_type, num_gpu)

    tags = ["relevant"]
    train_config.num_deletion = num_deletion
    train_config.g_val = float(g_val)
    train_config.drop_thres = float(drop_thres)

    train_self_explain(hp, train_config, save_dir,
                       data, data_loader, tags, modeling_option,
                       init_fn, informative_fn)

