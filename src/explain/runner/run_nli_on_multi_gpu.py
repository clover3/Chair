import sys

from data_generator.shared_setting import BertNLI
from explain.train_nli import get_nli_data, train_nli_multi_gpu
from models.transformer import hyperparams
from tf_util.tf_logging import set_level_debug
from trainer.model_saver import load_model_w_scope, load_model


def train_nil_from(model_path, save_dir, resume=False):
    print("Load model path : ", model_path)
    print("Save dir : ", save_dir)
    def load_fn(sess, model_path):
        if not resume:
            return load_model_w_scope(sess, model_path, "bert")
        else:
            return load_model(sess, model_path)

    steps = 67000
    hp = hyperparams.HPSENLI3()
    nli_setting = BertNLI()
    data = get_nli_data(hp, nli_setting)
    set_level_debug()
    hp = hyperparams.HPSENLI3()
    n_gpu = 2
    return train_nli_multi_gpu(hp, nli_setting, save_dir, steps, data, model_path, load_fn, n_gpu)


if __name__  == "__main__":
    train_nil_from(sys.argv[1], sys.argv[2])