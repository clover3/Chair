import sys

from data_generator.shared_setting import BertNLI
from explain.train_nli import get_nli_data, train_nli_multi_gpu
from models.transformer import hyperparams
from trainer.model_saver import load_model_w_scope


def train_nil_from(save_dir, model_path):
    def load_fn(sess, model_path):
        return load_model_w_scope(sess, model_path, "bert")

    steps = 67000
    hp = hyperparams.HPSENLI3()
    nli_setting = BertNLI()
    data = get_nli_data(hp, nli_setting)
    hp = hyperparams.HPSENLI3()
    n_gpu = 4
    return train_nli_multi_gpu(hp, nli_setting, save_dir, steps, data, model_path, load_fn, n_gpu)

if __name__  == "__main__":
    train_nil_from(sys.argv[1], sys.argv[2])