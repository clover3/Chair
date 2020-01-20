import sys

from explain.train_nli import train_nil_from
from trainer.model_saver import load_model_w_scope


def train_nil_from_bert(save_dir, model_path):
    def load_fn(sess, model_path):
        return load_model_w_scope(sess, model_path, "bert")
    max_steps = 61358
    return train_nil_from(save_dir, model_path, load_fn, max_steps)


if __name__  == "__main__":
    train_nil_from_bert(sys.argv[1], sys.argv[2])