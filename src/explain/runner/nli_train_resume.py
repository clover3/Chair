import sys

from cpath import get_latest_model_path_from_dir_path
from explain.train_nli import train_nil_from
from trainer.model_saver import load_model


def train_nil_resume(save_dir, steps):
    def load_fn(sess, model_path):
        return load_model(sess, model_path)

    model_path = get_latest_model_path_from_dir_path(save_dir)

    return train_nil_from(save_dir, model_path, load_fn, steps)


if __name__  == "__main__":
    train_nil_resume(sys.argv[1], int(sys.argv[2]))